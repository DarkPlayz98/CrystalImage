from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import Dataset
from .model import DarkImage
from .tokenizer import Tokenizer


DATA_DIR = Path("data/train")
CHECKPOINT_DIR = Path("checkpoints")
SAMPLE_DIR = Path("samples")

MODEL_PATH = CHECKPOINT_DIR / "crystal_image_v0_6.npz"
VOCAB_PATH = CHECKPOINT_DIR / "crystal_image_v0_6_vocab.json"


def save_sample(model, tokenizer, caption, step):
    text = tokenizer.text_vector(
        caption,
        model.embedding,
    )

    rng = np.random.default_rng(1000 + step)

    image = rng.normal(
        0,
        1,
        model.output_size,
    ).astype(np.float32)

    # Iterative denoising.
    for i in range(60):
        timestep = 1.0 - (
            i / 59.0
        )

        prediction = model.denoise(
            image,
            text,
            timestep,
        )

        strength = 0.12

        image = (
            (1.0 - strength) * image
            + strength * prediction
        )

        image = np.clip(
            image,
            0,
            1,
        )

    image = image.reshape(
        model.image_size,
        model.image_size,
        3,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        SAMPLE_DIR /
        f"v0_6_step_{step}.png"
    )

    Image.fromarray(
        (image * 255).astype(np.uint8),
        "RGB",
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST,
    ).save(output)

    return output


def main():
    print("================================")
    print("       Crystal Image v0.6")
    print("  Dataset + Sample Generation")
    print("================================")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = Dataset(
        DATA_DIR,
        32,
    )

    dataset.summary()

    if len(dataset) == 0:
        raise RuntimeError(
            "No training pairs found."
        )

    tokenizer = Tokenizer()

    tokenizer.build(
        dataset.captions()
    )

    model = DarkImage(
        vocab_size=len(tokenizer.vocab),
        seed=123,
    )

    rng = np.random.default_rng(2026)

    learning_rate = 0.001
    steps = 5000

    print(
        f"Training pairs: {len(dataset)}"
    )

    print(
        f"Vocabulary: {len(tokenizer.vocab)}"
    )

    print(
        f"Training steps: {steps}"
    )

    for step in range(steps):
        index = rng.integers(
            len(dataset)
        )

        target, caption, filename = (
            dataset.get(index)
        )

        text_vector = tokenizer.text_vector(
            caption,
            model.embedding,
        )

        timestep = (
            rng.integers(1, 101) / 100.0
        )

        noise = rng.normal(
            0,
            1,
            target.shape,
        ).astype(np.float32)

        noisy = (
            (1.0 - timestep) * target
            + timestep * noise
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

        d3 = (
            2.0 * error / error.size
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

        dh2 = d3 @ model.W3.T
        dh2[h2[0] <= 0] = 0

        dW2 = np.outer(
            h1[0],
            dh2,
        )

        db2 = dh2

        dh1 = dh2 @ model.W2.T
        dh1[h1[0] <= 0] = 0

        dW1 = np.outer(
            x[0],
            dh1,
        )

        db1 = dh1

        dx = dh1 @ model.W1.T

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
                d_text / len(token_ids)
            )

        gradients = [
            dW1,
            db1,
            dW2,
            db2,
            dW3,
            db3,
            d_embedding,
        ]

        for gradient in gradients:
            np.clip(
                gradient,
                -1.0,
                1.0,
                out=gradient,
            )

        model.W1 -= learning_rate * dW1
        model.b1 -= learning_rate * db1

        model.W2 -= learning_rate * dW2
        model.b2 -= learning_rate * db2

        model.W3 -= learning_rate * dW3
        model.b3 -= learning_rate * db3

        model.embedding -= (
            learning_rate *
            d_embedding
        )

        if (step + 1) % 250 == 0:
            print(
                f"step {step + 1:5d}/{steps} "
                f"loss={loss:.6f}"
            )

    # Save final checkpoint.
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

    # Generate an image from the trained model.
    sample_caption = dataset.get(0)[1]

    sample = save_sample(
        model,
        tokenizer,
        sample_caption,
        steps,
    )

    report = CHECKPOINT_DIR / "training_report.txt"

    report.write_text(
        "Crystal Image v0.6\n"
        "==================\n"
        f"Training pairs: {len(dataset)}\n"
        f"Vocabulary: {len(tokenizer.vocab)}\n"
        f"Steps: {steps}\n"
        f"Sample caption: {sample_caption}\n"
        f"Sample: {sample}\n",
        encoding="utf-8",
    )

    print()
    print("Training complete.")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Sample: {sample}")


if __name__ == "__main__":
    main()
