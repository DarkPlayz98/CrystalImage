from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 32000

    context_length: int = 512

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8

    ffn_multiplier: int = 3

    dropout: float = 0.0

    batch_size_per_gpu: int = 4
    gradient_accumulation: int = 8

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    warmup_steps: int = 250
    max_steps: int = 5000

    weight_decay: float = 0.1

    log_every: int = 10
    save_every: int = 500

    seed: int = 2026

    @property
    def ffn_dim(self):
        return self.d_model * self.ffn_multiplier
