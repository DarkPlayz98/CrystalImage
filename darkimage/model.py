from pathlib import Path
import numpy as np


class CrystalImage:
    def __init__(self):
        self.version = "1.0"
        self.center_x = 0.5
        self.center_y = 0.53
        self.scale = 0.95
        self.glow = 1.0
        self.sharpness = 1.0
        self.samples = 0

    def update(
        self,
        center_x,
        center_y,
        scale,
    ):
        n = self.samples

        self.center_x = (
            self.center_x * n + center_x
        ) / (n + 1)

        self.center_y = (
            self.center_y * n + center_y
        ) / (n + 1)

        self.scale = (
            self.scale * n + scale
        ) / (n + 1)

        self.samples += 1

    def save(self, path):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            version=self.version,
            center_x=self.center_x,
            center_y=self.center_y,
            scale=self.scale,
            glow=self.glow,
            sharpness=self.sharpness,
            samples=self.samples,
        )

    @classmethod
    def load(cls, path):
        data = np.load(
            path,
            allow_pickle=True,
        )

        model = cls()

        model.center_x = float(data["center_x"])
        model.center_y = float(data["center_y"])
        model.scale = float(data["scale"])
        model.glow = float(data["glow"])
        model.sharpness = float(
            data["sharpness"]
        )
        model.samples = int(data["samples"])

        return model
