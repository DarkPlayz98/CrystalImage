import numpy as np
import json
import re
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
        }

        if vocab:
            self.vocab.update(vocab)

    @staticmethod
    def tokenize(text):
        return re.findall(
            r"[a-z0-9']+",
            text.lower(),
        )

    def build(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        tokens = self.tokenize(text)

        ids = [
            self.vocab.get(
                token,
                self.vocab["<unk>"],
            )
            for token in tokens
        ]

        return ids or [self.vocab["<unk>"]]

    def text_vector(self, text, embedding):
        ids = self.encode(text)

        return embedding[ids].mean(
            axis=0
        ).astype(np.float32)

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                self.vocab,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        vocab = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return cls(vocab)
