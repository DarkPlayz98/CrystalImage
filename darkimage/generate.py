from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .model import DarkImage
from .tokenizer import Tokenizer


MODEL_PATH = (
    Path("checkpoints") /
    "darkimage_v0_5.npz"
)

VOCAB_PATH = (
    Path("checkpoints") /
    "darkimage_v0_5_vocab.json"
)

OUTPUT_DIR = (
    Path.home() /
    "storage" /
    "shared" /
    "Windows"
)


def main():
    prompt = " ".join(
        sys.argv[1:]
    ).strip()

    if not prompt:
        raise SystemExit(
            'Usage: python -m '
            'darkimage.generate '
            '"a black cat"'
        )

    if not MODEL_PATH.exists():
        raise SystemExit(
            "v0.5 model not found. "
            "Run training first."
        )

    tokenizer = Tokenizer.load(
        VOCAB_PATH
    )

    data = np.load(
        MODEL_PATH
    )

    model = DarkImage(
        vocab_size=len(
            tokenizer.vocab
        )
    )

    model.embedding = data["embedding"]
    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]
    model.W3 = data["W3"]
    model.b3 = data["b3"]

    text = tokenizer.text_vector(
        prompt,
        model.embedding,
    )

    rng = np.random.default_rng()

    # Start with pure noise.
    image = rng.normal(
        0,
        1,
        model.output_size,
    ).astype(np.float32)

    # Iteratively denoise.
    steps = 50

    for i in range(steps):
        timestep = (
            1.0 -
            (i / (steps - 1))
        )

        prediction = model.denoise(
            image,
            text,
            timestep,
        )

        # Gradually move toward prediction.
        strength = 0.12

        image = (
            (1.0 - strength) * image
            +
            strength * prediction
        )

        image = np.clip(
            image,
            0,
            1,
        )

    image = image.reshape(
        model.image_size,
        model.image_size,
        3,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = "_".join(
        prompt.lower().split()
    )[:50]

    output = (
        OUTPUT_DIR /
        f"darkimage_v0_5_{safe_name}.png"
    )

    Image.fromarray(
        (
            image * 255
        ).astype(np.uint8),
        "RGB",
    ).save(output)

    print(
        f"Generated: {output}"
    )


if __name__ == "__main__":
    main()
