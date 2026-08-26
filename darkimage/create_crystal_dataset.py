from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random

OUT = Path("data/train")
SIZE = 128
COUNT = 100

OUT.mkdir(parents=True, exist_ok=True)

for i in range(1, COUNT + 1):
    random.seed(1000 + i)

    img = Image.new("RGB", (SIZE, SIZE), (2, 4, 10))

    # Glow layer
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)

    for r in range(45, 2, -2):
        a = max(2, int(35 * (1 - r / 45)))
        g.ellipse(
            (64-r, 67-r, 64+r, 67+r),
            fill=(40, 150, 255, a),
        )

    glow = glow.filter(ImageFilter.GaussianBlur(12))
    img = Image.alpha_composite(img.convert("RGBA"), glow)

    # Crystal
    crystal = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(crystal)

    top = (64, 12)
    lt = (37, 45)
    lb = (44, 101)
    bottom = (64, 119)
    rb = (84, 101)
    rt = (91, 45)

    # Main body
    d.polygon(
        [top, rt, rb, bottom, lb, lt],
        fill=(30, 130, 230, 255),
    )

    # Facets
    d.polygon(
        [top, lt, lb, bottom],
        fill=(15, 75, 160, 255),
    )

    d.polygon(
        [top, (68, 43), (70, 100), bottom],
        fill=(80, 205, 255, 255),
    )

    d.polygon(
        [(68, 43), rt, rb, bottom, (70, 100)],
        fill=(25, 110, 205, 255),
    )

    # Top facet
    d.polygon(
        [top, lt, (68, 43), rt],
        fill=(130, 230, 255, 255),
    )

    # Sharp edges
    edge = (205, 250, 255, 255)

    d.line([top, lt, lb, bottom], fill=edge, width=2)
    d.line([top, rt, rb, bottom], fill=edge, width=2)
    d.line(
        [top, (68, 43), (70, 100), bottom],
        fill=(235, 255, 255, 255),
        width=2,
    )

    img = Image.alpha_composite(img, crystal)

    # Particles
    particles = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    p = ImageDraw.Draw(particles)

    for _ in range(100):
        x = random.randint(8, 119)
        y = random.randint(8, 119)

        if 34 < x < 94 and 8 < y < 121:
            continue

        r = random.choice([1, 1, 1, 2])
        p.ellipse(
            (x-r, y-r, x+r, y+r),
            fill=(100, 210, 255, random.randint(90, 220)),
        )

    img = Image.alpha_composite(img, particles)

    path = OUT / f"crystal_{i:04d}.png"
    img.convert("RGB").save(path, quality=95)

    (OUT / f"crystal_{i:04d}.txt").write_text(
        "a highly detailed glowing blue crystal floating in darkness, "
        "sharp geometric facets, luminous edges, bright magical core, "
        "cinematic lighting, realistic reflections, ultra detailed",
        encoding="utf-8",
    )

print(f"Created {COUNT} {SIZE}x{SIZE} crystal training pairs.")
