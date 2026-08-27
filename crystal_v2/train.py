from pathlib import Path
import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import CrystalImageV2


DATA = Path("data/train")
OUT = Path("v2_checkpoints")

IMAGE_SIZE = 512

AE_EPOCHS = 20
DIFFUSION_EPOCHS = 80

BATCH_SIZE = 4

LR = 2e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def tokenize(text):
    return re.findall(
        r"[a-z0-9']+",
        text.lower(),
    )


def build_vocab(captions):
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for caption in captions:
        for word in tokenize(
            caption
        ):
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def encode(
    text,
    vocab,
):
    ids = [
        vocab.get(
            word,
            vocab["<unk>"],
        )
        for word in tokenize(text)
    ]

    return ids or [1]


def load_data():
    pairs = []

    for image_path in sorted(
        DATA.glob("*.png")
    ):
        caption_path = image_path.with_suffix(
            ".txt"
        )

        if not caption_path.exists():
            continue

        caption = caption_path.read_text(
            encoding="utf-8"
        ).strip()

        if not caption:
            continue

        pairs.append(
            (
                image_path,
                caption,
            )
        )

    return pairs


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.LANCZOS,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 127.5 - 1.0

    return torch.from_numpy(
        array.transpose(2, 0, 1)
    )


def batchify(
    pairs,
    vocab,
    start,
):
    batch = pairs[
        start:start + BATCH_SIZE
    ]

    images = torch.stack(
        [
            load_image(path)
            for path, _ in batch
        ]
    )

    ids = [
        encode(
            caption,
            vocab,
        )
        for _, caption in batch
    ]

    max_len = max(
        len(x)
        for x in ids
    )

    tokens = torch.zeros(
        len(ids),
        max_len,
        dtype=torch.long,
    )

    for i, item in enumerate(ids):
        tokens[
            i,
            :len(item)
        ] = torch.tensor(
            item,
            dtype=torch.long,
        )

    return images, tokens


def main():
    print("=" * 50)
    print("       CRYSTAL IMAGE v2.0")
    print("       LATENT DIFFUSION")
    print("=" * 50)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = load_data()

    if not pairs:
        raise RuntimeError(
            "No training pairs found."
        )

    vocab = build_vocab(
        [
            caption
            for _, caption in pairs
        ]
    )

    model = CrystalImageV2(
        vocab_size=len(vocab)
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(
            f"Using {torch.cuda.device_count()} GPUs"
        )

        model = torch.nn.DataParallel(
            model
        )

    print(
        f"Training pairs: {len(pairs)}"
    )

    print(
        f"Vocabulary: {len(vocab)}"
    )

    print(
        f"GPU count: "
        f"{torch.cuda.device_count()}"
    )

    # ----------------------------
    # Stage 1: Autoencoder
    # ----------------------------

    print("\nStage 1: Autoencoder")

    ae_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    for epoch in range(
        1,
        AE_EPOCHS + 1,
    ):
        order = np.random.permutation(
            len(pairs)
        )

        total = 0.0

        for start in range(
            0,
            len(order),
            BATCH_SIZE,
        ):
            selected = [
                pairs[i]
                for i in order[
                    start:start + BATCH_SIZE
                ]
            ]

            images, _ = batchify(
                selected,
                vocab,
                0,
            )

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):
                reconstructed = model(
                    images
                ) if not isinstance(
                    model,
                    torch.nn.DataParallel
                ) else model.module(
                    images
                )

                loss = F.mse_loss(
                    reconstructed,
                    images,
                )

            ae_optimizer.zero_grad(
                set_to_none=True
            )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                ae_optimizer
            )

            scaler.update()

            total += loss.item()

        average = (
            total /
            max(
                1,
                len(pairs) / BATCH_SIZE
            )
        )

        print(
            f"AE epoch "
            f"{epoch}/{AE_EPOCHS} "
            f"loss={average:.6f}"
        )

    ae_path = (
        OUT /
        "crystal_image_v2_0_autoencoder.pt"
    )

    base_model = (
        model.module
        if isinstance(
            model,
            torch.nn.DataParallel,
        )
        else model
    )

    torch.save(
        {
            "autoencoder":
                base_model.autoencoder.state_dict(),
        },
        ae_path,
    )

    # ----------------------------
    # Stage 2: Latent diffusion
    # ----------------------------

    print("\nStage 2: Latent diffusion")

    # Freeze autoencoder.
    for parameter in (
        base_model.autoencoder.parameters()
    ):
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter
            in base_model.diffusion.parameters()
            if parameter.requires_grad
        ],
        lr=LR,
    )

    for epoch in range(
        1,
        DIFFUSION_EPOCHS + 1,
    ):
        order = np.random.permutation(
            len(pairs)
        )

        total = 0.0

        for start in range(
            0,
            len(order),
            BATCH_SIZE,
        ):
            selected = [
                pairs[i]
                for i in order[
                    start:start + BATCH_SIZE
                ]
            ]

            images, tokens = batchify(
                selected,
                vocab,
                0,
            )

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            tokens = tokens.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.no_grad():
                latent = (
                    base_model
                    .autoencoder
                    .encode(images)
                )

            noise = torch.randn_like(
                latent
            )

            timestep = torch.rand(
                latent.shape[0],
                device=DEVICE,
            )

            t = timestep.view(
                -1,
                1,
                1,
                1,
            )

            noisy = (
                latent * (1.0 - t)
                +
                noise * t
            )

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):
                predicted = (
                    base_model
                    .diffusion(
                        noisy,
                        tokens,
                        timestep,
                    )
                )

                loss = F.mse_loss(
                    predicted,
                    noise,
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            total += loss.item()

        average = (
            total /
            max(
                1,
                len(pairs) / BATCH_SIZE
            )
        )

        print(
            f"Diffusion epoch "
            f"{epoch}/{DIFFUSION_EPOCHS} "
            f"loss={average:.6f}"
        )

    final_path = (
        OUT /
        "crystal_image_v2_0.pt"
    )

    torch.save(
        {
            "version": "2.0",
            "vocab": vocab,
            "autoencoder":
                base_model.autoencoder.state_dict(),
            "diffusion":
                base_model.diffusion.state_dict(),
        },
        final_path,
    )

    (
        OUT /
        "crystal_image_v2_0_vocab.json"
    ).write_text(
        json.dumps(
            vocab,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Crystal Image v2.0 training complete."
    )
    print(
        f"Model: {final_path}"
    )


if __name__ == "__main__":
    main()
