from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("data/train")
SIZE = 512
COUNT = 150

CAPTION = (
    "a magnificent glowing crystal floating in deep darkness, "
    "sharp geometric facets, luminous blue energy, brilliant core, "
    "cinematic magical lighting, detailed reflections"
)

OUT.mkdir(parents=True, exist_ok=True)


def make_crystal(index):
    rng = random.Random(10000 + index)

    img = Image.new("RGB", (SIZE, SIZE), (2, 4, 12))

    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)

    cx = 256 + rng.randint(-20, 20)
    cy = 270 + rng.randint(-10, 20)

    for r in range(220, 10, -8):
        strength = max(1, int(35 * (1 - r / 220)))
        g.ellipse(
            (cx-r, cy-r, cx+r, cy+r),
            fill=(30, 130, 255, strength),
        )

    glow = glow.filter(ImageFilter.GaussianBlur(45))
    img = Image.alpha_composite(img.convert("RGBA"), glow)

    crystal = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(crystal)

    scale = rng.uniform(0.85, 1.12)

    top = (cx, int(cy - 210 * scale))
    lt = (int(cx - 125 * scale), int(cy - 80 * scale))
    lb = (int(cx - 90 * scale), int(cy + 170 * scale))
    bottom = (cx, int(cy + 220 * scale))
    rb = (int(cx + 90 * scale), int(cy + 170 * scale))
    rt = (int(cx + 125 * scale), int(cy - 80 * scale))

    center_top = (
        int(cx + 15 * scale),
        int(cy - 85 * scale),
    )

    center_low = (
        int(cx + 25 * scale),
        int(cy + 155 * scale),
    )

    d.polygon(
        [top, rt, rb, bottom, lb, lt],
        fill=(25, 115, 220, 255),
    )

    d.polygon(
        [top, lt, lb, bottom],
        fill=(8, 45, 125, 255),
    )

    d.polygon(
        [top, center_top, center_low, bottom],
        fill=(70, 205, 255, 255),
    )

    d.polygon(
        [center_top, rt, rb, bottom, center_low],
        fill=(15, 90, 190, 255),
    )

    d.polygon(
        [top, lt, center_top, rt],
        fill=(130, 235, 255, 255),
    )

    # Internal facets
    d.line(
        [lt, center_top, lb],
        fill=(100, 220, 255, 200),
        width=3,
    )

    d.line(
        [rt, center_top, rb],
        fill=(130, 235, 255, 200),
        width=3,
    )

    d.line(
        [top, lt, lb, bottom],
        fill=(220, 255, 255, 255),
        width=4,
    )

    d.line(
        [top, rt, rb, bottom],
        fill=(220, 255, 255, 255),
        width=4,
    )

    d.line(
        [top, center_top, center_low, bottom],
        fill=(245, 255, 255, 255),
        width=3,
    )

    img = Image.alpha_composite(img, crystal)

    particles = Image.new(
        "RGBA",
        (SIZE, SIZE),
        (0, 0, 0, 0),
    )
    p = ImageDraw.Draw(particles)

    for _ in range(180):
        x = rng.randint(10, SIZE - 10)
        y = rng.randint(10, SIZE - 10)

        if abs(x - cx) < 150 and abs(y - cy) < 230:
            continue

        r = rng.choice([1, 1, 2, 2, 3])
        p.ellipse(
            (x-r, y-r, x+r, y+r),
            fill=(
                100,
                rng.randint(180, 255),
                255,
                rng.randint(80, 220),
            ),
        )

    img = Image.alpha_composite(img, particles)

    return img.convert("RGB")


for i in range(1, COUNT + 1):
    name = f"crystal_{i:04d}"

    image = make_crystal(i)
    image.save(
        OUT / f"{name}.png",
        quality=95,
    )

    (OUT / f"{name}.txt").write_text(
        CAPTION,
        encoding="utf-8",
    )

print(
    f"Created {COUNT} "
    f"{SIZE}x{SIZE} training pairs."
)
