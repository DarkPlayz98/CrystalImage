from pathlib import Path

import numpy as np
from PIL import Image

from .model import DarkImage
from .tokenizer import Tokenizer


DATA_DIR = Path("data/train")
CHECKPOINT_DIR = Path("checkpoints")

MODEL_PATH = (
    CHECKPOINT_DIR /
    "darkimage_v0_5.npz"
)

VOCAB_PATH = (
    CHECKPOINT_DIR /
    "darkimage_v0_5_vocab.json"
)


def load_dataset(size):
    pairs = []

    for image_path in sorted(
        DATA_DIR.iterdir()
    ):
        if image_path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
        }:
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
            continue

        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize((size, size))
        )

        array = (
            np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0
        )

        pairs.append(
            (
                array.reshape(-1),
                caption,
            )
        )

    if not pairs:
        raise RuntimeError(
            "No image/caption pairs found."
        )

    return pairs


def main():
    print("================================")
    print("       DarkImage v0.5")
    print("   Iterative Denoising Model")
    print("================================")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = load_dataset(32)

    tokenizer = Tokenizer()

    tokenizer.build(
        [
            caption
            for _, caption in pairs
        ]
    )

    model = DarkImage(
        vocab_size=len(tokenizer.vocab),
        seed=123,
    )

    rng = np.random.default_rng(2026)

    learning_rate = 0.001
    steps = 5000

    print(
        f"Training pairs: {len(pairs)}"
    )

    print(
        f"Vocabulary: "
        f"{len(tokenizer.vocab)}"
    )

    print(
        f"Training steps: {steps}"
    )

    for step in range(steps):

        target, caption = pairs[
            rng.integers(len(pairs))
        ]

        text_vector = (
            tokenizer.text_vector(
                caption,
                model.embedding,
            )
        )

        # Random diffusion timestep.
        timestep = (
            rng.integers(1, 101)
            / 100.0
        )

        # More noise at larger timesteps.
        noise = rng.normal(
            0,
            1,
            target.shape,
        ).astype(np.float32)

        noisy = (
            (1.0 - timestep) * target
            +
            timestep * noise
        )

        prediction, x, h1, h2 = (
            model.forward(
                noisy[None, :],
                text_vector[None, :],
                timestep,
            )
        )

        prediction = prediction[0]

        error = prediction - target

        loss = np.mean(
            error ** 2
        )

        # Backpropagation.
        d3 = (
            2.0 *
            error /
            error.size
        )

        d3 *= (
            prediction *
            (1.0 - prediction)
        )

        dW3 = np.outer(
            h2[0],
            d3,
        )

        db3 = d3

        dh2 = (
            d3 @ model.W3.T
        )

        dh2[h2[0] <= 0] = 0

        dW2 = np.outer(
            h1[0],
            dh2,
        )

        db2 = dh2

        dh1 = (
            dh2 @ model.W2.T
        )

        dh1[h1[0] <= 0] = 0

        dW1 = np.outer(
            x[0],
            dh1,
        )

        db1 = dh1

        dx = (
            dh1 @ model.W1.T
        )

        d_text = dx[
            model.output_size:
            model.output_size +
            model.embedding_size
        ]

        token_ids = tokenizer.encode(
            caption
        )

        d_embedding = np.zeros_like(
            model.embedding
        )

        for token_id in token_ids:
            d_embedding[token_id] += (
                d_text /
                len(token_ids)
            )

        # Gradient clipping keeps CPU training
        # from exploding on difficult examples.
        for gradient in [
            dW1,
            db1,
            dW2,
            db2,
            dW3,
            db3,
            d_embedding,
        ]:
            np.clip(
                gradient,
                -1.0,
                1.0,
                out=gradient,
            )

        model.W1 -= (
            learning_rate * dW1
        )

        model.b1 -= (
            learning_rate * db1
        )

        model.W2 -= (
            learning_rate * dW2
        )

        model.b2 -= (
            learning_rate * db2
        )

        model.W3 -= (
            learning_rate * dW3
        )

        model.b3 -= (
            learning_rate * db3
        )

        model.embedding -= (
            learning_rate *
            d_embedding
        )

        if (step + 1) % 250 == 0:
            print(
                f"step {step + 1:5d}/"
                f"{steps} "
                f"loss={loss:.6f}"
            )

    np.savez_compressed(
        MODEL_PATH,
        embedding=model.embedding,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        W3=model.W3,
        b3=model.b3,
    )

    tokenizer.save(
        VOCAB_PATH
    )

    print()
    print("Training complete.")
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Vocabulary: {VOCAB_PATH}"
    )


if __name__ == "__main__":
    main()
