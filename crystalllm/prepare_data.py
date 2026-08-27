from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


OUT = Path("data/corpus/train.txt")

MAX_STORIES = 300000


def main():
    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading TinyStories stream..."
    )

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    count = 0

    with OUT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for item in tqdm(
            dataset,
            total=MAX_STORIES,
        ):
            text = item["text"].strip()

            if len(text) < 20:
                continue

            handle.write(
                text.replace(
                    "\n",
                    " ",
                )
            )

            handle.write("\n")

            count += 1

            if count >= MAX_STORIES:
                break

    print()
    print(
        f"Wrote {count} stories."
    )
    print(
        f"Corpus: {OUT}"
    )


if __name__ == "__main__":
    main()
