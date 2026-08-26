import numpy as np


class CrystalImage:
    """
    Crystal Image v0.8
    64x64 text-conditioned denoising model.

    This is a lightweight NumPy prototype designed to run
    in GitHub Actions and on Termux without PyTorch.
    """

    VERSION = "0.8"
    IMAGE_SIZE = 64
    CHANNELS = 3

    def __init__(
        self,
        vocab_size=64,
        embedding_dim=32,
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.rng = np.random.default_rng(seed)

        self.image_dim = (
            self.IMAGE_SIZE
            * self.IMAGE_SIZE
            * self.CHANNELS
        )

        # Compact latent representation.
        self.latent_size = 256

        self.embedding = (
            self.rng.normal(
                0,
                0.02,
                (
                    vocab_size,
                    embedding_dim,
                ),
            ).astype(np.float32)
        )

        self.text_projection = (
            self.rng.normal(
                0,
                0.02,
                (
                    embedding_dim,
                    self.latent_size,
                ),
            ).astype(np.float32)
        )

        self.image_projection = (
            self.rng.normal(
                0,
                0.02,
                (
                    self.image_dim,
                    self.latent_size,
                ),
            ).astype(np.float32)
        )

        self.output_projection = (
            self.rng.normal(
                0,
                0.02,
                (
                    self.latent_size,
                    self.image_dim,
                ),
            ).astype(np.float32)
        )

        self.bias = np.zeros(
            self.image_dim,
            dtype=np.float32,
        )

    def forward(
        self,
        noisy,
        text_vector,
        timestep,
    ):
        noisy = np.asarray(
            noisy,
            dtype=np.float32,
        )

        text_vector = np.asarray(
            text_vector,
            dtype=np.float32,
        )

        if noisy.ndim == 1:
            noisy = noisy[None, :]

        if text_vector.ndim == 1:
            text_vector = text_vector[None, :]

        batch = noisy.shape[0]

        image_features = (
            noisy @ self.image_projection
        )

        text_features = (
            text_vector @ self.text_projection
        )

        # Normalize timestep.
        t = np.asarray(
            timestep,
            dtype=np.float32,
        )

        if t.ndim == 0:
            t = np.full(
                (batch, 1),
                float(t) / 1000.0,
                dtype=np.float32,
            )
        elif t.ndim == 1:
            t = t.reshape(-1, 1) / 1000.0

        # Time conditioning.
        time_features = np.repeat(
            t,
            self.latent_size,
            axis=1,
        )

        hidden = (
            image_features
            + text_features
            + time_features * 0.1
        )

        hidden = np.tanh(hidden)

        prediction = (
            hidden @ self.output_projection
            + self.bias
        )

        return prediction

    def save(self, path):
        np.savez(
            path,
            version=self.VERSION,
            image_size=self.IMAGE_SIZE,
            channels=self.CHANNELS,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            latent_size=self.latent_size,
            embedding=self.embedding,
            text_projection=self.text_projection,
            image_projection=self.image_projection,
            output_projection=self.output_projection,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path):
        data = np.load(
            path,
            allow_pickle=False,
        )

        model = cls(
            vocab_size=int(
                data["vocab_size"]
            ),
            embedding_dim=int(
                data["embedding_dim"]
            ),
        )

        model.embedding = data["embedding"]
        model.text_projection = data[
            "text_projection"
        ]
        model.image_projection = data[
            "image_projection"
        ]
        model.output_projection = data[
            "output_projection"
        ]
        model.bias = data["bias"]

        return model
