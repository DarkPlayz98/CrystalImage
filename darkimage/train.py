from pathlib import Path

import numpy as np

from .dataset import Dataset
from .model import CrystalImage
from .tokenizer import Tokenizer


DATA = Path("data/train")
CHECKPOINTS = Path("checkpoints")

SIZE = 64
STEPS = 50000
SAVE_EVERY = 1000


def main():
    print("================================")
    print("       Crystal Image v0.8")
    print("        64x64 Denoiser")
    print("================================")

    CHECKPOINTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = Dataset(
        str(DATA),
        image_size=SIZE,
    )

    pairs = dataset.pairs

    if not pairs:
        raise RuntimeError(
            "No image/caption pairs found in data/train"
        )

    captions = [
        caption
        for _, caption in pairs
    ]

    # Build vocabulary using the actual
    # Tokenizer API.
    tokenizer = Tokenizer()
    tokenizer.build(captions)

    vocab_size = len(tokenizer.vocab)

    model = CrystalImage(
        vocab_size=vocab_size
    )

    print(
        f"Training pairs: {len(pairs)}"
    )

    print(
        f"Vocabulary: {vocab_size}"
    )

    print(
        f"Resolution: {SIZE}x{SIZE}"
    )

    print(
        f"Training steps: {STEPS}"
    )

    rng = np.random.default_rng(
        20260826
    )

    for step in range(
        1,
        STEPS + 1,
    ):
        image_path, caption = pairs[
            (step - 1) % len(pairs)
        ]

        image = dataset.load_image(
            image_path
        )

        clean = (
            image.astype(np.float32)
            / 127.5
            - 1.0
        ).reshape(-1)

        text_vector = tokenizer.text_vector(
            caption,
            model.embedding,
        )

        timestep = int(
            rng.integers(
                1,
                1000,
            )
        )

        noise = rng.normal(
            0,
            1,
            clean.shape,
        ).astype(np.float32)

        strength = (
            timestep / 1000.0
        )

        noisy = (
            clean * (1.0 - strength)
            + noise * strength
        )

        prediction = model.forward(
            noisy[None, :],
            text_vector[None, :],
            timestep,
        )[0]

        error = prediction - noise

        loss = float(
            np.mean(error * error)
        )

        learning_rate = 0.000001

        hidden = np.tanh(
            noisy @ model.image_projection
            + text_vector
            @ model.text_projection
            + (timestep / 1000.0) * 0.1
        )

        model.output_projection -= (
            learning_rate
            * np.outer(
                hidden,
                error,
            )
        )

        model.bias -= (
            learning_rate * error
        )

        if step % 100 == 0:
            print(
                f"step {step}/{STEPS} "
                f"loss={loss:.6f}"
            )

        if step % SAVE_EVERY == 0:
            checkpoint = (
                CHECKPOINTS
                / f"crystal_image_v0_8_step_{step}.npz"
            )

            model.save(checkpoint)

            print(
                f"Saved checkpoint: {checkpoint}"
            )

    final = (
        CHECKPOINTS
        / "crystal_image_v0_8.npz"
    )

    model.save(final)

    # Save vocabulary beside the model.
    tokenizer.save(
        CHECKPOINTS
        / "crystal_image_v0_8_vocab.json"
    )

    print(
        f"Saved final checkpoint: {final}"
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
