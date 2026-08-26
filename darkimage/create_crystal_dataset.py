from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("data/train")
SIZE = 64
COUNT = 100
SCALE = 4

COLORS = [
    (40, 150, 255),
    (130, 70, 255),
    (30, 230, 180),
    (255, 60, 130),
    (80, 220, 255),
    (255, 175, 45),
]


def make_crystal(rng, color):
    s = SIZE * SCALE

    img = Image.new("RGB", (s, s), (1, 2, 6))

    # Background stars/particles
    draw = ImageDraw.Draw(img)

    for _ in range(rng.integers(15, 35)):
        x = rng.integers(0, s)
        y = rng.integers(0, s)
        r = rng.choice([1, 1, 2, 3])

        draw.ellipse(
            (x-r, y-r, x+r, y+r),
            fill=(80, 90, 120),
        )

    cx = s // 2 + rng.integers(-12, 13)
    cy = s // 2 + rng.integers(-8, 10)

    w = rng.integers(22, 34) * SCALE
    h = rng.integers(34, 50) * SCALE

    # Soft controlled glow
    glow = Image.new(
        "RGBA",
        (s, s),
        (0, 0, 0, 0),
    )

    gd = ImageDraw.Draw(glow)

    gd.ellipse(
        (
            cx - w,
            cy - h,
            cx + w,
            cy + h,
        ),
        fill=(*color, 80),
    )

    glow = glow.filter(
        ImageFilter.GaussianBlur(10 * SCALE)
    )

    img = Image.alpha_composite(
        img.convert("RGBA"),
        glow,
    )

    # Crystal geometry
    top = (cx, cy - h // 2)
    upper_left = (cx - w // 2, cy - h // 5)
    lower_left = (cx - w // 2, cy + h // 3)
    bottom = (cx, cy + h // 2)
    lower_right = (cx + w // 2, cy + h // 3)
    upper_right = (cx + w // 2, cy - h // 5)

    pts = [
        top,
        upper_right,
        lower_right,
        bottom,
        lower_left,
        upper_left,
    ]

    crystal = Image.new(
        "RGBA",
        (s, s),
        (0, 0, 0, 0),
    )

    d = ImageDraw.Draw(crystal)

    # Main crystal
    d.polygon(
        pts,
        fill=(*color, 235),
    )

    # Left facet
    d.polygon(
        [
            top,
            (cx, bottom[1]),
            lower_left,
            upper_left,
        ],
        fill=(
            min(color[0] + 35, 255),
            min(color[1] + 35, 255),
            min(color[2] + 35, 255),
            245,
        ),
    )

    # Bright center facet
    d.polygon(
        [
            top,
            (cx + w // 8, cy - h // 8),
            (cx + w // 5, cy + h // 3),
            bottom,
        ],
        fill=(
            min(color[0] + 100, 255),
            min(color[1] + 100, 255),
            min(color[2] + 100, 255),
            250,
        ),
    )

    # Dark right facet
    d.polygon(
        [
            top,
            upper_right,
            lower_right,
            bottom,
        ],
        fill=(
            max(color[0] - 40, 0),
            max(color[1] - 40, 0),
            max(color[2] - 40, 0),
            235,
        ),
    )

    # Internal facets
    edge = (
        min(color[0] + 110, 255),
        min(color[1] + 110, 255),
        min(color[2] + 110, 255),
        220,
    )

    lw = max(2, SCALE)

    d.line(
        [top, bottom],
        fill=edge,
        width=lw,
    )

    d.line(
        [upper_left, bottom],
        fill=edge,
        width=lw,
    )

    d.line(
        [upper_right, bottom],
        fill=edge,
        width=lw,
    )

    # Sharp outer edge
    d.line(
        pts + [pts[0]],
        fill=(255, 255, 255, 230),
        width=lw,
        joint="curve",
    )

    # Bright core
    d.ellipse(
        (
            cx - w // 10,
            cy - h // 5,
            cx + w // 10,
            cy + h // 5,
        ),
        fill=(255, 255, 255, 85),
    )

    img = Image.alpha_composite(
        img,
        crystal,
    )

    # Slight high-resolution glow integration
    glow2 = crystal.filter(
        ImageFilter.GaussianBlur(2 * SCALE)
    )

    img = Image.alpha_composite(
        img,
        glow2,
    )

    # Keep the crystal sharp.
    img = img.convert("RGB").resize(
        (SIZE, SIZE),
        Image.Resampling.LANCZOS,
    )

    return img


def caption_for(color):
    names = {
        (40, 150, 255): "blue",
        (130, 70, 255): "purple",
        (30, 230, 180): "green",
        (255, 60, 130): "pink",
        (80, 220, 255): "cyan",
        (255, 175, 45): "golden",
    }

    return (
        f"a detailed glowing {names[color]} crystal "
        "floating in darkness, sharp geometric facets, "
        "luminous edges, magical energy, bright core"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for p in OUT.glob("crystal_*.png"):
        p.unlink()

    for p in OUT.glob("crystal_*.txt"):
        p.unlink()

    rng = np.random.default_rng(20260826)

    for i in range(1, COUNT + 1):
        color = COLORS[
            rng.integers(len(COLORS))
        ]

        image = make_crystal(rng, color)

        image.save(
            OUT / f"crystal_{i:04d}.png"
        )

        (
            OUT / f"crystal_{i:04d}.txt"
        ).write_text(
            caption_for(color),
            encoding="utf-8",
        )

    print(
        f"Created {COUNT} 64x64 crystal training pairs."
    )


if __name__ == "__main__":
    main()
