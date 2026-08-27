from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import CrystalTokenizer


CORPUS = Path(
    "data/corpus/train.txt"
)

TOKENIZER = Path(
    "checkpoints/crystal_tokenizer.model"
)

OUTPUT = Path(
    "data/tokens/train.bin"
)


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = CrystalTokenizer(
        TOKENIZER
    )

    all_ids = []

    with CORPUS.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in tqdm(
            handle,
            desc="Tokenizing",
        ):
            line = line.strip()

            if not line:
                continue

            ids = tokenizer.encode(
                line
            )

            all_ids.extend(ids)

    array = np.asarray(
        all_ids,
        dtype=np.uint16,
    )

    array.tofile(
        OUTPUT
    )

    print(
        f"Tokens: {len(array):,}"
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
