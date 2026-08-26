import numpy as np


class CrystalImage:
    """
    Crystal Image v0.7

    Tiny fully custom 32x32 text-conditioned
    encoder/decoder model.

    No pretrained weights.
    """

    def __init__(
        self,
        vocab_size,
        image_size=32,
        embedding_size=32,
        latent_size=128,
        hidden_size=256,
        seed=42,
    ):
        self.image_size = image_size
        self.output_size = image_size * image_size * 3
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        self.embedding = (
            rng.normal(
                0,
                0.05,
                (vocab_size, embedding_size),
            )
        ).astype(np.float32)

        # Image encoder.
        self.W_encode = (
            rng.normal(
                0,
                np.sqrt(
                    2.0 / self.output_size
                ),
                (
                    self.output_size,
                    latent_size,
                ),
            )
        ).astype(np.float32)

        self.b_encode = np.zeros(
            latent_size,
            dtype=np.float32,
        )

        # Text + latent conditioning.
        condition_size = (
            latent_size +
            embedding_size +
            1
        )

        self.W_condition = (
            rng.normal(
                0,
                np.sqrt(
                    2.0 / condition_size
                ),
                (
                    condition_size,
                    hidden_size,
                ),
            )
        ).astype(np.float32)

        self.b_condition = np.zeros(
            hidden_size,
            dtype=np.float32,
        )

        # Decoder.
        self.W_decode = (
            rng.normal(
                0,
                np.sqrt(
                    2.0 / hidden_size
                ),
                (
                    hidden_size,
                    self.output_size,
                ),
            )
        ).astype(np.float32)

        self.b_decode = np.zeros(
            self.output_size,
            dtype=np.float32,
        )

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (
            1.0 +
            np.exp(
                -np.clip(x, -30, 30)
            )
        )

    def encode(self, image):
        image = np.asarray(
            image,
            dtype=np.float32,
        ).reshape(1, -1)

        latent = self.relu(
            image @ self.W_encode
            + self.b_encode
        )

        return latent[0]

    def condition(
        self,
        latent,
        text_vector,
        timestep=0.0,
    ):
        x = np.concatenate(
            [
                np.asarray(
                    latent,
                    dtype=np.float32,
                ).reshape(-1),

                np.asarray(
                    text_vector,
                    dtype=np.float32,
                ).reshape(-1),

                np.asarray(
                    [timestep],
                    dtype=np.float32,
                ),
            ]
        )

        hidden = self.relu(
            x @ self.W_condition
            + self.b_condition
        )

        return hidden, x

    def decode(self, hidden):
        output = self.sigmoid(
            hidden @ self.W_decode
            + self.b_decode
        )

        return output

    def forward(
        self,
        image,
        text_vector,
        timestep=0.0,
    ):
        latent = self.encode(image)

        hidden, condition_input = (
            self.condition(
                latent,
                text_vector,
                timestep,
            )
        )

        output = self.decode(hidden)

        return (
            output,
            latent,
            hidden,
            condition_input,
        )

    def generate(
        self,
        text_vector,
        steps=50,
        seed=1234,
    ):
        rng = np.random.default_rng(
            seed
        )

        # Start in latent space.
        latent = rng.normal(
            0,
            1,
            self.latent_size,
        ).astype(np.float32)

        # Iterative latent denoising.
        for i in range(steps):
            timestep = 1.0 - (
                i / max(steps - 1, 1)
            )

            hidden, _ = self.condition(
                latent,
                text_vector,
                timestep,
            )

            prediction = self.decode(
                hidden
            )

            # Compress prediction back toward
            # latent dimensionality using the
            # encoder weights.
            predicted_latent = (
                prediction @ self.W_encode
            )

            predicted_latent = np.tanh(
                predicted_latent
            )

            strength = 0.08

            latent = (
                (1.0 - strength) * latent
                +
                strength * predicted_latent
            )

        hidden, _ = self.condition(
            latent,
            text_vector,
            0.0,
        )

        return self.decode(hidden)
