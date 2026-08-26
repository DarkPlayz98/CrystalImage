from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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


def make_crystal_base(seed=20260826):
    rng = np.random.default_rng(seed)

    img = Image.new(
        "RGB",
        (SIZE, SIZE),
        (3, 5, 12),
    )

    glow = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )

    gd = ImageDraw.Draw(glow)

    # Large soft blue glow
    for radius in range(22, 2, -2):
        alpha = int(
            2 + (22 - radius) * 1.5
        )

        gd.ellipse(
            (
                32 - radius,
                34 - radius,
                32 + radius,
                34 + radius,
            ),
            fill=(40, 150, 255, alpha),
        )

    glow = glow.filter(
        ImageFilter.GaussianBlur(7)
    )

    img = Image.alpha_composite(
        img.convert("RGBA"),
        glow,
    )

    crystal = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )

    cd = ImageDraw.Draw(crystal)

    # Main crystal silhouette
    top = (32, 7)
    left_top = (18, 23)
    left_bottom = (22, 51)
    bottom = (32, 59)
    right_bottom = (43, 51)
    right_top = (47, 23)

    cd.polygon(
        [
            top,
            right_top,
            right_bottom,
            bottom,
            left_bottom,
            left_top,
        ],
        fill=(35, 150, 245, 245),
    )

    # Left facet
    cd.polygon(
        [
            top,
            left_top,
            left_bottom,
            bottom,
        ],
        fill=(18, 95, 180, 245),
    )

    # Bright center facet
    cd.polygon(
        [
            top,
            (34, 22),
            (35, 49),
            bottom,
            (28, 49),
            (29, 22),
        ],
        fill=(70, 200, 255, 245),
    )

    # Right facet
    cd.polygon(
        [
            (34, 22),
            right_top,
            right_bottom,
            bottom,
            (35, 49),
        ],
        fill=(20, 115, 205, 245),
    )

    # Bright top
    cd.polygon(
        [
            top,
            left_top,
            (34, 22),
            right_top,
        ],
        fill=(115, 225, 255, 255),
    )

    # Sharp luminous edges
    edge = (170, 245, 255, 255)

    cd.line(
        [top, left_top, left_bottom, bottom],
        fill=edge,
        width=1,
    )

    cd.line(
        [top, right_top, right_bottom, bottom],
        fill=edge,
        width=1,
    )

    cd.line(
        [top, (34, 22), (35, 49), bottom],
        fill=(220, 250, 255, 255),
        width=1,
    )

    img = Image.alpha_composite(
        img,
        crystal,
    )

    # Magical particles
    particles = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )

    pd = ImageDraw.Draw(particles)

    for _ in range(35):
        x = int(rng.integers(6, 58))
        y = int(rng.integers(6, 58))

        # Keep particles away from the main crystal.
        if 15 < x < 49 and 5 < y < 61:
            continue

        r = int(rng.integers(1, 2))

        pd.ellipse(
            (x-r, y-r, x+r, y+r),
            fill=(100, 210, 255, 170),
        )

    img = Image.alpha_composite(
        img,
        particles,
    )

    return np.asarray(
        img.convert("RGB"),
        dtype=np.float32,
    )


def main():
    print("==============================")
    print("     Crystal Image v0.8")
    print("       64x64 Generator")
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

    print(f"Prompt: {prompt}")

    # Start with a structured crystal image,
    # then allow the trained model to influence it.
    base = make_crystal_base()

    image = (
        base / 127.5
        - 1.0
    ).reshape(-1)

    text = tokenizer.text_vector(
        prompt,
        model.embedding,
    )

    print("Applying model refinement...")

    for timestep in range(
        999,
        19,
        -20,
    ):
        prediction = model.forward(
            image[None, :],
            text[None, :],
            timestep,
        )[0]

        # Small refinement so the model does
        # not immediately destroy the structure.
        image -= prediction * 0.0005

        if timestep % 100 == 0:
            print(
                f"timestep {timestep}"
            )

    image = np.clip(
        image,
        -1,
        1,
    )

    image = (
        (image + 1.0) * 127.5
    ).astype(np.uint8)

    image = image.reshape(
        SIZE,
        SIZE,
        3,
    )

    Image.fromarray(
        image,
        "RGB",
    ).save(OUTPUT)

    print()
    print(
        f"Generated: {OUTPUT}"
    )
    print(
        "Crystal image generation complete."
    )


if __name__ == "__main__":
    main()
