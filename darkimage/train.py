from pathlib import Path
import json
import re

import numpy as np
import torch
from PIL import Image


DATA = Path("data/train")
CHECKPOINTS = Path("checkpoints")

IMAGE_SIZE = 96

EPOCHS = 80
BATCH_SIZE = 4
LEARNING_RATE = 2e-4

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

    for text in captions:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    words = tokenize(text)

    ids = [
        vocab.get(
            word,
            vocab["<unk>"],
        )
        for word in words
    ]

    if not ids:
        ids = [vocab["<unk>"]]

    return ids


def load_dataset():
    pairs = []

    for image_path in sorted(
        DATA.glob("*.png")
    ):
        text_path = image_path.with_suffix(
            ".txt"
        )

        if not text_path.exists():
            continue

        caption = text_path.read_text(
            encoding="utf-8"
        ).strip()

        if not caption:
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.LANCZOS,
            )

            array = (
                np.asarray(
                    img,
                    dtype=np.float32,
                )
                / 255.0
            )

        pairs.append(
            (
                array,
                caption,
            )
        )

    if not pairs:
        raise RuntimeError(
            "No training pairs found."
        )

    return pairs


def main():
    print("=" * 45)
    print("       Crystal Image v1.2")
    print("       1536x1536 Target")
    print("=" * 45)

    torch.manual_seed(2026)
    np.random.seed(2026)

    pairs = load_dataset()

    captions = [
        caption
        for _, caption in pairs
    ]

    vocab = build_vocab(
        captions
    )

    from .model import CrystalImage

    model = CrystalImage(
        vocab_size=len(vocab)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = torch.nn.MSELoss()

    print(
        f"Training pairs: {len(pairs)}"
    )

    print(
        f"Vocabulary: {len(vocab)}"
    )

    print(
        f"Training resolution: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Target resolution: 1536x1536"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        order = np.random.permutation(
            len(pairs)
        )

        total_loss = 0.0

        for start in range(
            0,
            len(order),
            BATCH_SIZE,
        ):
            batch_ids = order[
                start:
                start + BATCH_SIZE
            ]

            images = []
            captions_batch = []

            for index in batch_ids:
                image, caption = pairs[
                    index
                ]

                images.append(
                    image.transpose(
                        2,
                        0,
                        1,
                    )
                )

                captions_batch.append(
                    encode(
                        caption,
                        vocab,
                    )
                )

            max_len = max(
                len(ids)
                for ids in captions_batch
            )

            token_tensor = torch.zeros(
                (
                    len(captions_batch),
                    max_len,
                ),
                dtype=torch.long,
                device=DEVICE,
            )

            for i, ids in enumerate(
                captions_batch
            ):
                token_tensor[
                    i,
                    :len(ids),
                ] = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=DEVICE,
                )

            clean = torch.tensor(
                np.stack(images),
                dtype=torch.float32,
                device=DEVICE,
            )

            # Normalize to [-1, 1].
            clean = clean * 2.0 - 1.0

            noise = torch.randn_like(
                clean
            )

            timestep = torch.rand(
                (
                    clean.shape[0],
                    1,
                ),
                device=DEVICE,
            )

            noisy = (
                clean * (1.0 - timestep.view(
                    -1,
                    1,
                    1,
                    1,
                ))
                +
                noise * timestep.view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            # Predict clean image.
            prediction = model(
                noisy,
                token_tensor,
                timestep,
            )

            loss = loss_fn(
                prediction,
                clean,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_ids)
            )

        average = (
            total_loss /
            len(pairs)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"epoch {epoch:03d}/{EPOCHS} "
                f"loss={average:.6f}"
            )

    CHECKPOINTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "version": "1.2",
            "image_size": IMAGE_SIZE,
            "target_size": 1536,
            "vocab": vocab,
            "model": model.state_dict(),
        },
        CHECKPOINTS /
        "crystal_image_v1_2.pt",
    )

    (
        CHECKPOINTS /
        "crystal_image_v1_2_vocab.json"
    ).write_text(
        json.dumps(
            vocab,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Crystal Image v1.2 training complete."
    )


if __name__ == "__main__":
    main()
