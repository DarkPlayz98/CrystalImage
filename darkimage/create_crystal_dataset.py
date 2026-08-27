from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


OUT = Path("data/train")

# The model trains at a compact resolution.
# The source images are higher resolution.
SOURCE_SIZE = 512
COUNT = 120


COLORS = [
    (45, 155, 255),
    (115, 70, 255),
    (35, 230, 180),
    (255, 70, 145),
    (75, 215, 255),
    (255, 180, 55),
]


def create_image(seed):
    rng = random.Random(seed)

    img = Image.new(
        "RGB",
        (SOURCE_SIZE, SOURCE_SIZE),
        (2, 4, 12),
    )

    color = rng.choice(COLORS)

    cx = SOURCE_SIZE // 2 + rng.randint(-35, 35)
    cy = SOURCE_SIZE // 2 + rng.randint(-25, 25)

    width = rng.randint(105, 155)
    height = rng.randint(220, 330)

    # Background glow.
    glow = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    gd = ImageDraw.Draw(glow)

    for radius in range(220, 15, -10):
        alpha = max(
            2,
            int(
                42 *
                (1.0 - radius / 220.0)
            ),
        )

        gd.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(
                color[0],
                color[1],
                color[2],
                alpha,
            ),
        )

    glow = glow.filter(
        ImageFilter.GaussianBlur(35)
    )

    img = Image.alpha_composite(
        img.convert("RGBA"),
        glow,
    )

    # Crystal layer.
    crystal = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    d = ImageDraw.Draw(crystal)

    top = (cx, cy - height // 2)
    left_top = (
        cx - width // 2,
        cy - height // 5,
    )
    left_bottom = (
        cx - width // 2,
        cy + height // 3,
    )
    bottom = (
        cx,
        cy + height // 2,
    )
    right_bottom = (
        cx + width // 2,
        cy + height // 3,
    )
    right_top = (
        cx + width // 2,
        cy - height // 5,
    )

    center_top = (
        cx + rng.randint(-8, 8),
        cy - height // 7,
    )

    center_bottom = (
        cx + rng.randint(-8, 8),
        cy + height // 3,
    )

    points = [
        top,
        right_top,
        right_bottom,
        bottom,
        left_bottom,
        left_top,
    ]

    # Main body.
    d.polygon(
        points,
        fill=(
            color[0],
            color[1],
            color[2],
            245,
        ),
    )

    # Left facet.
    d.polygon(
        [
            top,
            left_top,
            left_bottom,
            bottom,
        ],
        fill=(
            max(color[0] - 25, 0),
            max(color[1] - 55, 0),
            max(color[2] - 20, 0),
            250,
        ),
    )

    # Center facet.
    d.polygon(
        [
            top,
            center_top,
            center_bottom,
            bottom,
        ],
        fill=(
            min(color[0] + 65, 255),
            min(color[1] + 65, 255),
            min(color[2] + 65, 255),
            255,
        ),
    )

    # Right facet.
    d.polygon(
        [
            center_top,
            right_top,
            right_bottom,
            bottom,
            center_bottom,
        ],
        fill=(
            max(color[0] - 15, 0),
            max(color[1] - 30, 0),
            max(color[2] - 15, 0),
            245,
        ),
    )

    # Top facet.
    d.polygon(
        [
            top,
            left_top,
            center_top,
            right_top,
        ],
        fill=(
            min(color[0] + 80, 255),
            min(color[1] + 80, 255),
            min(color[2] + 80, 255),
            255,
        ),
    )

    # Interior edges.
    edge = (
        210,
        250,
        255,
        230,
    )

    d.line(
        [top, left_top, left_bottom, bottom],
        fill=edge,
        width=2,
    )

    d.line(
        [top, right_top, right_bottom, bottom],
        fill=edge,
        width=2,
    )

    d.line(
        [top, center_top, center_bottom, bottom],
        fill=(245, 255, 255, 245),
        width=2,
    )

    # Energy core.
    core_width = rng.randint(7, 13)

    d.ellipse(
        (
            cx - core_width,
            cy - height // 6,
            cx + core_width,
            cy + height // 6,
        ),
        fill=(255, 255, 255, 90),
    )

    img = Image.alpha_composite(
        img,
        crystal,
    )

    # Floating energy particles.
    particles = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    pd = ImageDraw.Draw(particles)

    for _ in range(100):
        x = rng.randint(8, SOURCE_SIZE - 8)
        y = rng.randint(8, SOURCE_SIZE - 8)

        if (
            abs(x - cx) < width
            and abs(y - cy) < height
        ):
            continue

        radius = rng.choice(
            [1, 1, 2, 2, 3]
        )

        pd.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            fill=(
                min(color[0] + 70, 255),
                min(color[1] + 70, 255),
                255,
                rng.randint(80, 210),
            ),
        )

    img = Image.alpha_composite(
        img,
        particles,
    )

    return img.convert("RGB")


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in OUT.glob("crystal_*.png"):
        path.unlink()

    for path in OUT.glob("crystal_*.txt"):
        path.unlink()

    caption = (
        "a magnificent glowing crystal "
        "floating in darkness, sharp crystalline "
        "facets, luminous edges, magical blue energy, "
        "brilliant inner core, cinematic lighting, "
        "detailed reflections"
    )

    for i in range(1, COUNT + 1):
        image = create_image(
            12000 + i
        )

        image.save(
            OUT / f"crystal_{i:04d}.png",
            quality=95,
        )

        (
            OUT / f"crystal_{i:04d}.txt"
        ).write_text(
            caption,
            encoding="utf-8",
        )

    print(
        f"Created {COUNT} training pairs."
    )


if __name__ == "__main__":
    main()
