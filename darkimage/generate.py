from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter

from .model import CrystalImage


CHECKPOINT = Path(
    "checkpoints/crystal_image_v1_0.npz"
)

OUTPUT = Path(
    "generated_crystal_v1_0.png"
)

SIZE = 512


def main():
    print("=" * 40)
    print("       Crystal Image v1.0")
    print("       512x512 Generator")
    print("=" * 40)

    model = CrystalImage.load(
        CHECKPOINT
    )

    rng = random.Random(5122026)

    cx = int(
        SIZE * model.center_x
    )
    cy = int(
        SIZE * model.center_y
    )

    scale = model.scale

    img = Image.new(
        "RGB",
        (SIZE, SIZE),
        (2, 4, 12),
    )

    # Atmospheric glow
    glow = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )

    gd = ImageDraw.Draw(glow)

    max_radius = int(
        220 * model.glow
    )

    for r in range(
        max_radius,
        10,
        -8,
    ):
        alpha = max(
            1,
            int(
                30 * (
                    1 - r / max_radius
                )
            ),
        )

        gd.ellipse(
            (
                cx-r,
                cy-r,
                cx+r,
                cy+r,
            ),
            fill=(
                25,
                130,
                255,
                alpha,
            ),
        )

    glow = glow.filter(
        ImageFilter.GaussianBlur(45)
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

    d = ImageDraw.Draw(crystal)

    top = (cx, int(cy - 210 * scale))
    lt = (
        int(cx - 125 * scale),
        int(cy - 80 * scale),
    )
    lb = (
        int(cx - 90 * scale),
        int(cy + 170 * scale),
    )
    bottom = (
        cx,
        int(cy + 220 * scale),
    )
    rb = (
        int(cx + 90 * scale),
        int(cy + 170 * scale),
    )
    rt = (
        int(cx + 125 * scale),
        int(cy - 80 * scale),
    )

    ct = (
        int(cx + 15 * scale),
        int(cy - 85 * scale),
    )

    cl = (
        int(cx + 25 * scale),
        int(cy + 155 * scale),
    )

    d.polygon(
        [top, rt, rb, bottom, lb, lt],
        fill=(20, 110, 225, 255),
    )

    d.polygon(
        [top, lt, lb, bottom],
        fill=(5, 35, 115, 255),
    )

    d.polygon(
        [top, ct, cl, bottom],
        fill=(80, 215, 255, 255),
    )

    d.polygon(
        [ct, rt, rb, bottom, cl],
        fill=(10, 85, 190, 255),
    )

    d.polygon(
        [top, lt, ct, rt],
        fill=(150, 240, 255, 255),
    )

    edge_width = max(
        2,
        int(3 * model.sharpness),
    )

    edge = (
        220,
        255,
        255,
        255,
    )

    d.line(
        [top, lt, lb, bottom],
        fill=edge,
        width=edge_width,
    )

    d.line(
        [top, rt, rb, bottom],
        fill=edge,
        width=edge_width,
    )

    d.line(
        [top, ct, cl, bottom],
        fill=(245, 255, 255, 255),
        width=edge_width,
    )

    d.line(
        [lt, ct, lb],
        fill=(100, 220, 255, 200),
        width=edge_width,
    )

    d.line(
        [rt, ct, rb],
        fill=(130, 235, 255, 200),
        width=edge_width,
    )

    img = Image.alpha_composite(
        img,
        crystal,
    )

    # Energy particles
    particles = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )

    pd = ImageDraw.Draw(particles)

    for _ in range(220):
        x = rng.randint(10, SIZE - 10)
        y = rng.randint(10, SIZE - 10)

        if (
            abs(x - cx) < 150
            and abs(y - cy) < 230
        ):
            continue

        r = rng.choice([1, 2, 2, 3])

        pd.ellipse(
            (
                x-r,
                y-r,
                x+r,
                y+r,
            ),
            fill=(
                100,
                rng.randint(180, 255),
                255,
                rng.randint(80, 220),
            ),
        )

    img = Image.alpha_composite(
        img,
        particles,
    )

    img.convert("RGB").save(
        OUTPUT,
        quality=98,
    )

    print(
        f"Generated: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
