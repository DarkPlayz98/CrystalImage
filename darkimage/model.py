import numpy as np


class DarkImage:
    """
    Tiny text-conditioned denoising generator.

    Built from random initialization.
    No pretrained model is used.
    """

    def __init__(
        self,
        vocab_size,
        image_size=32,
        embedding_size=32,
        hidden_size=256,
        seed=42,
    ):
        self.image_size = image_size
        self.output_size = image_size * image_size * 3
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        # Our own word embeddings.
        self.embedding = (
            rng.normal(
                0,
                0.05,
                (vocab_size, embedding_size),
            )
        ).astype(np.float32)

        # Input = noisy image + text + timestep.
        input_size = (
            self.output_size +
            embedding_size +
            1
        )

        self.W1 = (
            rng.normal(
                0,
                np.sqrt(2.0 / input_size),
                (input_size, hidden_size),
            )
        ).astype(np.float32)

        self.b1 = np.zeros(
            hidden_size,
            dtype=np.float32,
        )

        self.W2 = (
            rng.normal(
                0,
                np.sqrt(2.0 / hidden_size),
                (hidden_size, hidden_size),
            )
        ).astype(np.float32)

        self.b2 = np.zeros(
            hidden_size,
            dtype=np.float32,
        )

        self.W3 = (
            rng.normal(
                0,
                np.sqrt(2.0 / hidden_size),
                (hidden_size, self.output_size),
            )
        ).astype(np.float32)

        self.b3 = np.zeros(
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

    def forward(
        self,
        noisy_image,
        text_vector,
        timestep,
    ):
        noisy_image = np.asarray(
            noisy_image,
            dtype=np.float32,
        ).reshape(1, -1)

        text_vector = np.asarray(
            text_vector,
            dtype=np.float32,
        ).reshape(1, -1)

        timestep_array = np.asarray(
            [[timestep]],
            dtype=np.float32,
        )

        x = np.concatenate(
            [
                noisy_image,
                text_vector,
                timestep_array,
            ],
            axis=1,
        )

        h1 = self.relu(
            x @ self.W1 + self.b1
        )

        h2 = self.relu(
            h1 @ self.W2 + self.b2
        )

        output = self.sigmoid(
            h2 @ self.W3 + self.b3
        )

        return output, x, h1, h2

    def denoise(
        self,
        image,
        text_vector,
        timestep,
    ):
        output, _, _, _ = self.forward(
            image,
            text_vector,
            timestep,
        )

        return output[0]
