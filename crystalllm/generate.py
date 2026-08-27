cat > crystalllm/generate.py <<'PY'
import argparse
from pathlib import Path

import torch

from .config import Config
from .model import CrystalLLM
from .tokenizer import CrystalTokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prompt"
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "crystalllm_step_5000.pt"
        ),
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = CrystalTokenizer(
        "checkpoints/"
        "crystal_tokenizer.model"
    )

    config = Config()
    config.vocab_size = (
        tokenizer.vocab_size
    )

    model = CrystalLLM(
        config
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    ids = tokenizer.encode(
        args.prompt,
        add_bos=True,
        add_eos=False,
    )

    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    output = model.generate(
        tokens,
        max_new_tokens=args.tokens,
        temperature=0.75,
        top_k=50,
    )

    print()
    print(
        tokenizer.decode(
            output[0].tolist()
        )
    )


if __name__ == "__main__":
    main()
