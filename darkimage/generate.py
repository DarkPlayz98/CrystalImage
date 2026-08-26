from pathlib import Path

import numpy as np
from PIL import Image

from .model import CrystalImage
from .tokenizer import Tokenizer


CHECKPOINT = Path(
    "checkpoints/crystal_image_v0_8.npz"
)

VOCAB = Path(
    "checkpoints/crystal_image_v0_8_vocab.json"
)

OUTPUT = Path(
    "generated_crystal_v0_8.png"
)

SIZE = 64


def main():
    print("==============================")
    print("   Crystal Image v0.8")
    print("      64x64 Generator")
    print("==============================")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    if not VOCAB.exists():
        raise FileNotFoundError(
            f"Missing vocabulary: {VOCAB}"
        )

    model = CrystalImage.load(
        CHECKPOINT
    )

    tokenizer = Tokenizer.load(
        VOCAB
    )

    prompt = (
        "a detailed glowing blue crystal "
        "floating in darkness, sharp geometric "
        "facets, luminous edges, magical energy, "
        "bright core"
    )

    text = tokenizer.text_vector(
        prompt,
        model.embedding,
    )

    rng = np.random.default_rng(
        20260826
    )

    # Start from random noise.
    image = rng.normal(
        0,
        1,
        model.image_dim,
    ).astype(np.float32)

    # Iterative denoising.
    for timestep in range(
        999,
        0,
        -20,
    ):
        prediction = model.forward(
            image[None, :],
            text[None, :],
            timestep,
        )[0]

        strength = 0.035

        image -= (
            prediction * strength
        )

        if timestep % 100 == 0:
            print(
                f"denoising timestep "
                f"{timestep}"
            )

    # Convert model output to RGB.
    image = np.clip(
        image,
        -1,
        1,
    )

    image = (
        (image + 1.0)
        * 127.5
    ).astype(np.uint8)

    image = image.reshape(
        SIZE,
        SIZE,
        3,
    )

    result = Image.fromarray(
        image,
        "RGB",
    )

    result.save(OUTPUT)

    print()
    print(
        f"Generated: {OUTPUT}"
    )

    print(
        f"Resolution: {SIZE}x{SIZE}"
    )


if __name__ == "__main__":
    main()
