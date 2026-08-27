from pathlib import Path
import re

import torch
from PIL import Image


CHECKPOINT = Path(
    "v2_checkpoints/crystal_image_v2_0.pt"
)

OUTPUT = Path(
    "crystal_image_v2_0.png"
)

LATENT_SIZE = 64
FINAL_SIZE = 1536


def tokenize(text):
    return re.findall(
        r"[a-z0-9']+",
        text.lower(),
    )


def main():
    from .model import CrystalImageV2

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    vocab = checkpoint["vocab"]

    model = CrystalImageV2(
        vocab_size=len(vocab)
    ).to(device)

    model.autoencoder.load_state_dict(
        checkpoint["autoencoder"]
    )

    model.diffusion.load_state_dict(
        checkpoint["diffusion"]
    )

    model.eval()

    prompt = (
        "a magnificent glowing blue crystal "
        "floating in darkness, sharp crystalline "
        "facets, luminous edges, magical energy, "
        "brilliant inner core, detailed reflections"
    )

    ids = [
        vocab.get(
            word,
            vocab["<unk>"],
        )
        for word in tokenize(prompt)
    ]

    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        latent = torch.randn(
            (
                1,
                4,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            device=device,
        )

        # Iterative DDIM-like denoising.
        for i in range(
            50,
            0,
            -1,
        ):
            t = torch.tensor(
                [i / 50.0],
                device=device,
            )

            noise_prediction = (
                model.diffusion(
                    latent,
                    tokens,
                    t,
                )
            )

            step_size = (
                1.0 / 50.0
            )

            latent = (
                latent
                -
                noise_prediction
                * step_size
            )

        image = model.autoencoder.decode(
            latent
        )

        image = (
            image.clamp(-1, 1)
            + 1
        ) * 127.5

        image = image[0].permute(
            1,
            2,
            0,
        ).byte().cpu().numpy()

    image = Image.fromarray(
        image,
        "RGB",
    )

    image = image.resize(
        (
            FINAL_SIZE,
            FINAL_SIZE,
        ),
        Image.Resampling.LANCZOS,
    )

    image.save(
        OUTPUT,
        quality=98,
    )

    print(
        f"Generated {OUTPUT}"
    )
    print(
        f"Resolution: "
        f"{FINAL_SIZE}x{FINAL_SIZE}"
    )


if __name__ == "__main__":
    main()
