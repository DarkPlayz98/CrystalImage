from .dataset import Dataset


def main():
    print("==============================")
    print("     Crystal Image Dataset")
    print("==============================")

    dataset = Dataset(
        "data/train",
        32,
    )

    dataset.summary()

    if len(dataset) == 0:
        raise RuntimeError(
            "No valid image/caption pairs."
        )

    print()
    print("Dataset check: PASS")


if __name__ == "__main__":
    main()
