from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image


CHECKPOINT = Path(
    "checkpoints/crystal_image_v1_2.pt"
)

OUTPUT = Path(
    "generated_crystal_v1_2.png"
)

LATENT_SIZE = 96
OUTPUT_SIZE = 1536


def tokenize(text):
    return re.findall(
        r"[a-z0-9']+",
        text.lower(),
    )


def main():
    from .model import CrystalImage

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    vocab = checkpoint["vocab"]

    model = CrystalImage(
        vocab_size=len(vocab)
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    prompt = (
        "a magnificent glowing crystal "
        "floating in darkness, sharp crystalline "
        "facets, luminous edges, magical blue energy, "
        "brilliant inner core, cinematic lighting, "
        "detailed reflections"
    )

    words = tokenize(prompt)

    ids = [
        vocab.get(
            word,
            vocab["<unk>"],
        )
        for word in words
    ]

    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    print(
        "Generating 96x96 latent image..."
    )

    with torch.no_grad():
        # Start from noise.
        image = torch.randn(
            (
                1,
                3,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            device=device,
        )

        # Iterative denoising.
        for step in range(
            60,
            0,
            -1,
        ):
            timestep = torch.tensor(
                [[step / 60.0]],
                dtype=torch.float32,
                device=device,
            )

            prediction = model(
                image,
                tokens,
                timestep,
            )

            strength = (
                0.04
                if step > 20
                else 0.02
            )

            image = (
                image * (1.0 - strength)
                +
                prediction * strength
            )

    image = image[
        0
    ].permute(
        1,
        2,
        0,
    )

    image = (
        image.clamp(-1, 1)
        + 1
    ) * 127.5

    image = image.byte().cpu().numpy()

    small = Image.fromarray(
        image,
        "RGB",
    )

    # High-quality 1536 reconstruction.
    large = small.resize(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
        ),
        Image.Resampling.LANCZOS,
    )

    large.save(
        OUTPUT,
        quality=98,
    )

    print(
        f"Generated: {OUTPUT}"
    )

    print(
        f"Resolution: "
        f"{OUTPUT_SIZE}x{OUTPUT_SIZE}"
    )


if __name__ == "__main__":
    main()
