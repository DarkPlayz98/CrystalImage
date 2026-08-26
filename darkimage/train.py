from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import Dataset
from .model import CrystalImage
from .tokenizer import Tokenizer


DATA_DIR = Path("data/train")
CHECKPOINT_DIR = Path("checkpoints")
SAMPLE_DIR = Path("samples")

STEPS = 5000
IMAGE_SIZE = 32
LEARNING_RATE = 0.0005


def save_image(vector, path):
    image = np.clip(
        vector.reshape(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        0,
        1,
    )

    Image.fromarray(
        (image * 255).astype(np.uint8),
        "RGB",
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST,
    ).save(path)


def main():
    print("================================")
    print("       Crystal Image v0.7")
    print("       32x32 Latent Model")
    print("================================")

    dataset = Dataset(
        DATA_DIR,
        IMAGE_SIZE,
    )

    dataset.summary()

    if len(dataset) == 0:
        raise RuntimeError(
            "No training pairs found."
        )

    tokenizer = Tokenizer()
    tokenizer.build(dataset.captions())

    model = CrystalImage(
        vocab_size=len(tokenizer.vocab),
        image_size=IMAGE_SIZE,
        seed=2026,
    )

    rng = np.random.default_rng(2026)

    print(f"Training pairs: {len(dataset)}")
    print(f"Vocabulary: {len(tokenizer.vocab)}")
    print(f"Training steps: {STEPS}")

    for step in range(STEPS):
        index = rng.integers(len(dataset))

        target, caption, _ = dataset.get(index)

        text = tokenizer.text_vector(
            caption,
            model.embedding,
        )

        # Train reconstruction with noise.
        noise_level = rng.uniform(
            0.05,
            0.35,
        )

        noise = rng.normal(
            0,
            1,
            target.shape,
        ).astype(np.float32)

        noisy = (
            (1.0 - noise_level) * target
            + noise_level * noise
        )

        noisy = np.clip(
            noisy,
            0,
            1,
        )

        prediction, latent, hidden, condition_input = (
            model.forward(
                noisy,
                text,
                noise_level,
            )
        )

        error = prediction - target
        loss = np.mean(error ** 2)

        # Decoder gradient.
        d_output = (
            2.0 * error / error.size
        )

        d_sigmoid = (
            prediction *
            (1.0 - prediction)
        )

        d_decode = (
            d_output * d_sigmoid
        )

        dW_decode = np.outer(
            hidden,
            d_decode,
        )

        db_decode = d_decode

        # Hidden layer gradient.
        dhidden = (
            d_decode @ model.W_decode.T
        )

        dhidden[hidden <= 0] = 0

        dW_condition = np.outer(
            condition_input,
            dhidden,
        )

        db_condition = dhidden

        dcondition = (
            dhidden @ model.W_condition.T
        )

        latent_grad = dcondition[
            :model.latent_size
        ]

        text_grad = dcondition[
            model.latent_size:
            model.latent_size +
            model.embedding_size
        ]

        # Encoder gradient.
        encoded_input = noisy

        dW_encode = np.outer(
            encoded_input,
            latent_grad,
        )

        db_encode = latent_grad

        # Update weights.
        np.clip(
            dW_decode,
            -1,
            1,
            out=dW_decode,
        )

        np.clip(
            dW_condition,
            -1,
            1,
            out=dW_condition,
        )

        np.clip(
            dW_encode,
            -1,
            1,
            out=dW_encode,
        )

        model.W_decode -= (
            LEARNING_RATE *
            dW_decode
        )

        model.b_decode -= (
            LEARNING_RATE *
            db_decode
        )

        model.W_condition -= (
            LEARNING_RATE *
            dW_condition
        )

        model.b_condition -= (
            LEARNING_RATE *
            db_condition
        )

        model.W_encode -= (
            LEARNING_RATE *
            dW_encode
        )

        model.b_encode -= (
            LEARNING_RATE *
            db_encode
        )

        # Update the words used by this caption.
        token_ids = tokenizer.encode(
            caption
        )

        if token_ids:
            for token_id in token_ids:
                model.embedding[token_id] -= (
                    LEARNING_RATE *
                    text_grad /
                    len(token_ids)
                )

        if (step + 1) % 250 == 0:
            print(
                f"step {step + 1:5d}/{STEPS} "
                f"loss={loss:.6f}"
            )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        CHECKPOINT_DIR /
        "crystal_image_v0_7.npz"
    )

    np.savez_compressed(
        checkpoint,
        embedding=model.embedding,
        W_encode=model.W_encode,
        b_encode=model.b_encode,
        W_condition=model.W_condition,
        b_condition=model.b_condition,
        W_decode=model.W_decode,
        b_decode=model.b_decode,
    )

    vocab = (
        CHECKPOINT_DIR /
        "crystal_image_v0_7_vocab.json"
    )

    tokenizer.save(vocab)

    # Generate a sample using the trained model.
    caption = dataset.get(0)[1]

    text = tokenizer.text_vector(
        caption,
        model.embedding,
    )

    generated = model.generate(
        text,
        steps=50,
        seed=2026,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = (
        SAMPLE_DIR /
        "crystal_image_v0_7.png"
    )

    save_image(
        generated,
        sample,
    )

    report = (
        CHECKPOINT_DIR /
        "crystal_image_v0_7_report.txt"
    )

    report.write_text(
        "Crystal Image v0.7\n"
        "==================\n"
        f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
        f"Training pairs: {len(dataset)}\n"
        f"Vocabulary: {len(tokenizer.vocab)}\n"
        f"Steps: {STEPS}\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Sample caption: {caption}\n",
        encoding="utf-8",
    )

    print()
    print("Training complete!")
    print(f"Checkpoint: {checkpoint}")
    print(f"Generated:  {sample}")


if __name__ == "__main__":
    main()
