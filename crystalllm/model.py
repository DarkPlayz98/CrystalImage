import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-6,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x


class RotaryEmbedding:
    def __init__(
        self,
        dim,
        max_seq_len,
        base=10000,
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        self.cos = freqs.cos()
        self.sin = freqs.sin()

    def to(self, device):
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        return self


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack(
        (-x2, x1),
        dim=-1,
    ).flatten(-2)


def apply_rope(
    x,
    cos,
    sin,
):
    seq_len = x.shape[-2]

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    cos = torch.repeat_interleave(
        cos,
        2,
        dim=-1,
    )

    sin = torch.repeat_interleave(
        sin,
        2,
        dim=-1,
    )

    return (
        x * cos
        + rotate_half(x) * sin
    )


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        hidden_dim,
    ):
        super().__init__()

        self.gate = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.up = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

    def forward(self, x):
        return self.down(
            F.silu(
                self.gate(x)
            )
            * self.up(x)
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        context_length,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(
            d_model,
            d_model * 3,
            bias=False,
        )

        self.out = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            context_length,
        )

    def forward(self, x):
        b, t, c = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        q = q.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        device = x.device

        self.rope.to(device)

        cos = self.rope.cos
        sin = self.rope.sin

        q = apply_rope(
            q,
            cos,
            sin,
        )

        k = apply_rope(
            k,
            cos,
            sin,
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        y = y.transpose(
            1,
            2,
        ).contiguous()

        y = y.view(
            b,
            t,
            c,
        )

        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            config.d_model
        )

        self.attn = (
            CausalSelfAttention(
                config.d_model,
                config.n_heads,
                config.context_length,
            )
        )

        self.norm2 = RMSNorm(
            config.d_model
        )

        self.mlp = SwiGLU(
            config.d_model,
            config.ffn_dim,
        )

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class CrystalLLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(
                    config.n_layers
                )
            ]
        )

        self.norm = RMSNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        tokens,
        targets=None,
    ):
        x = self.token_embedding(
            tokens
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens=150,
        temperature=0.8,
        top_k=50,
    ):
        self.eval()

        for _ in range(
            max_new_tokens
        ):
            context = tokens[
                :,
                -self.config.context_length:
            ]

            logits, _ = self(
                context
            )

            logits = logits[
                :,
                -1,
                :
            ]

            logits = logits / max(
                temperature,
                1e-5,
            )

            if top_k:
                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1),
                    ),
                )

                threshold = values[
                    :,
                    -1,
                    None,
                ]

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf"),
                )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [
                    tokens,
                    next_token,
                ],
                dim=1,
            )

        return tokens
