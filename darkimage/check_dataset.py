from pathlib import Path

from PIL import Image


DATASET = Path("data/train")
EXPECTED_SIZE = (128, 128)


def main():
    print("==============================")
    print("     Crystal Image Dataset")
    print("==============================")
    print(f"Dataset: {DATASET}")

    images = sorted(DATASET.glob("*.png"))

    if not images:
        raise RuntimeError(
            "No images found in data/train"
        )

    valid = 0
    pairs = 0

    for image_path in images:
        caption_path = image_path.with_suffix(".txt")

        if not caption_path.exists():
            print(
                f"Skipping {image_path.name}: "
                f"missing {caption_path.name}"
            )
            continue

        with Image.open(image_path) as image:
            size = image.size

        if size != EXPECTED_SIZE:
            print(
                f"Skipping {image_path.name}: "
                f"size is {size[0]}x{size[1]}, "
                f"expected 128x128"
            )
            continue

        valid += 1
        pairs += 1

    print(f"Images: {len(images)}")
    print(f"Valid training pairs: {pairs}")
    print("Resolution: 128x128")

    if pairs:
        example = next(
            DATASET.glob("*.png")
        )

        caption = example.with_suffix(".txt")

        print("Example:")
        print(f"  {example.name}")
        print(
            f'  "{caption.read_text(encoding="utf-8").strip()}"'
        )

    if pairs == 0:
        raise RuntimeError(
            "No valid 128x128 image/caption pairs found."
        )

    print()
    print("Dataset check: PASS")


if __name__ == "__main__":
    main()
