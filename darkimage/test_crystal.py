from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 64
OUT = Path("generated_crystal.png")


def main():
    rng = np.random.default_rng(20260826)

    y, x = np.mgrid[0:SIZE, 0:SIZE]

    # Dark background
    image = np.zeros(
        (SIZE, SIZE, 3),
        dtype=np.float32,
    )

    # Crystal position
    cx = 32
    cy = 34

    color = np.array(
        [50, 180, 255],
        dtype=np.float32,
    )

    # Soft atmospheric glow
    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    glow = np.exp(
        -(distance ** 2) / (2 * 18 ** 2)
    )

    image += glow[..., None] * color * 0.35

    # Crystal polygon
    points = np.array([
        [32, 7],
        [47, 22],
        [44, 52],
        [32, 59],
        [20, 52],
        [17, 22],
    ])

    # Point-in-polygon using PIL mask
    mask = Image.new(
        "L",
        (SIZE, SIZE),
        0,
    )

    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask)
    draw.polygon(
        [tuple(p) for p in points],
        fill=255,
    )

    mask_array = (
        np.asarray(mask)
        .astype(bool)
    )

    # Main crystal
    image[mask_array] = color

    # Facets
    facet1 = (
        mask_array &
        (x < 32)
    )

    image[facet1] *= 0.65

    facet2 = (
        mask_array &
        (x >= 32) &
        (y < 35)
    )

    image[facet2] *= 1.25

    facet2 = np.clip(
        image,
        0,
        255,
    )

    # Bright center
    core = (
        mask_array &
        (abs(x - 32) < 4) &
        (y > 18) &
        (y < 48)
    )

    image[core] += 80

    # Crystal edges
    edge = (
        (abs(x - 32) < 1.5) &
        mask_array
    )

    image[edge] = 255

    # Floating particles
    for _ in range(25):
        px = rng.integers(5, 59)
        py = rng.integers(5, 59)

        if not mask_array[py, px]:
            image[
                max(0, py-1):min(SIZE, py+2),
                max(0, px-1):min(SIZE, px+2)
            ] += color * 0.5

    image = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        image,
        "RGB",
    ).save(OUT)

    print(f"Created {OUT}")
    print(f"Resolution: {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
