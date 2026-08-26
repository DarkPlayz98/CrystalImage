from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


class Dataset:
    def __init__(
        self,
        directory="data/train",
        image_size=32,
    ):
        self.directory = Path(directory)
        self.image_size = image_size

        self.pairs = []

        self._load()

    def _load(self):
        if not self.directory.exists():
            raise RuntimeError(
                f"Dataset directory not found: "
                f"{self.directory}"
            )

        for image_path in sorted(
            self.directory.iterdir()
        ):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            caption_path = image_path.with_suffix(
                ".txt"
            )

            if not caption_path.exists():
                print(
                    f"Skipping {image_path.name}: "
                    f"missing caption"
                )
                continue

            caption = caption_path.read_text(
                encoding="utf-8"
            ).strip()

            if not caption:
                print(
                    f"Skipping {image_path.name}: "
                    f"empty caption"
                )
                continue

            try:
                image = (
                    Image.open(image_path)
                    .convert("RGB")
                    .resize(
                        (
                            self.image_size,
                            self.image_size,
                        )
                    )
                )

                array = (
                    np.asarray(
                        image,
                        dtype=np.float32,
                    )
                    / 255.0
                )

                self.pairs.append(
                    (
                        array.reshape(-1),
                        caption,
                        image_path.name,
                    )
                )

            except Exception as exc:
                print(
                    f"Skipping {image_path.name}: "
                    f"{exc}"
                )

    def __len__(self):
        return len(self.pairs)

    def get(self, index):
        return self.pairs[index]

    def captions(self):
        return [
            caption
            for _, caption, _ in self.pairs
        ]

    def summary(self):
        print(
            f"Dataset: {self.directory}"
        )
        print(
            f"Images: {len(self.pairs)}"
        )
        print(
            f"Resolution: "
            f"{self.image_size}x"
            f"{self.image_size}"
        )

        if self.pairs:
            print(
                "Example:"
            )
            print(
                f"  {self.pairs[0][2]}"
            )
            print(
                f"  \"{self.pairs[0][1]}\""
            )
