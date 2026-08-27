from pathlib import Path
from PIL import Image

DATA = Path("data/train")
SIZE = (512, 512)


def main():
    print("=" * 30)
    print(" Crystal Image v1.0 Dataset")
    print("=" * 30)

    if not DATA.exists():
        raise RuntimeError("data/train does not exist")

    valid = 0
    images = sorted(DATA.glob("*.png"))

    for image_path in images:
        text_path = image_path.with_suffix(".txt")

        if not text_path.exists():
            print(
                f"Skipping {image_path.name}: "
                "missing caption"
            )
            continue

        try:
            with Image.open(image_path) as img:
                if img.size != SIZE:
                    print(
                        f"Skipping {image_path.name}: "
                        f"size is {img.size}, "
                        "expected 512x512"
                    )
                    continue
        except Exception as error:
            print(
                f"Skipping {image_path.name}: {error}"
            )
            continue

        valid += 1

    print(f"Dataset: {DATA}")
    print(f"Images: {len(images)}")
    print(f"Valid training pairs: {valid}")
    print("Resolution: 512x512")

    if valid == 0:
        raise RuntimeError(
            "No valid 512x512 pairs found."
        )

    print()
    print("Dataset check: PASS")


if __name__ == "__main__":
    main()
