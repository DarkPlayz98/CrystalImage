from pathlib import Path
import time
import numpy as np
from PIL import Image

from .model import CrystalImage


DATA = Path("data/train")
CHECKPOINT = Path(
    "checkpoints/crystal_image_v1_0.npz"
)

SIZE = 512

# Long training
EPOCHS = 250


def analyze_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(
            img,
            dtype=np.float32,
        ) / 255.0

    brightness = arr.mean(axis=2)

    threshold = np.percentile(
        brightness,
        92,
    )

    mask = brightness >= threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 0.5, 0.53, 0.95

    center_x = float(xs.mean() / SIZE)
    center_y = float(ys.mean() / SIZE)

    spread_x = (
        xs.max() - xs.min()
    ) / SIZE

    scale = max(
        0.5,
        min(1.2, spread_x * 2.0),
    )

    return (
        center_x,
        center_y,
        scale,
    )


def main():
    print("=" * 40)
    print("       Crystal Image v1.0")
    print("     512x512 Long Training")
    print("=" * 40)

    images = sorted(
        DATA.glob("*.png")
    )

    if not images:
        raise RuntimeError(
            "No training images found"
        )

    model = CrystalImage()

    print(
        f"Training pairs: {len(images)}"
    )
    print(f"Epochs: {EPOCHS}")
    print("Mode: spatial parameter learning")
    print()

    started = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        for image_path in images:
            x, y, scale = analyze_image(
                image_path
            )

            model.update(
                x,
                y,
                scale,
            )

        # Slow refinement passes.
        model.glow = min(
            1.5,
            model.glow + 0.001,
        )

        model.sharpness = min(
            1.8,
            model.sharpness + 0.0015,
        )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            elapsed = int(
                time.time() - started
            )

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| center=({model.center_x:.3f}, "
                f"{model.center_y:.3f}) "
                f"| scale={model.scale:.3f} "
                f"| {elapsed}s"
            )

    model.save(CHECKPOINT)

    print()
    print("Training complete")
    print(
        f"Saved: {CHECKPOINT}"
    )


if __name__ == "__main__":
    main()
