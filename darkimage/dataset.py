from pathlib import Path

import numpy as np
from PIL import Image


class Dataset:
    def __init__(
        self,
        root,
        image_size=64,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.pairs = []

        for image_path in sorted(
            self.root.glob("*.png")
        ):
            caption_path = image_path.with_suffix(
                ".txt"
            )

            if not caption_path.exists():
                continue

            self.pairs.append(
                (
                    image_path,
                    caption_path.read_text(
                        encoding="utf-8"
                    ).strip(),
                )
            )

    def load_image(self, path):
        image = Image.open(path).convert(
            "RGB"
        )

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            image,
            dtype=np.float32,
        )
