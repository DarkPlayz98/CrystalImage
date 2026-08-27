import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

    def forward(self, x):
        h = self.conv1(
            F.silu(
                self.norm1(x)
            )
        )

        h = self.conv2(
            F.silu(
                self.norm2(h)
            )
        )

        return x + h


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.proj = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        mask = (
            tokens != 0
        ).float().unsqueeze(-1)

        x = x * mask

        denom = mask.sum(
            dim=1
        ).clamp_min(1)

        x = x.sum(
            dim=1
        ) / denom

        return self.proj(x)


class AutoEncoder(nn.Module):
    """
    512x512 image -> 64x64 latent -> 512x512 image
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.SiLU(),

            nn.Conv2d(
                128,
                256,
                4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                256,
            ),
            nn.SiLU(),

            nn.Conv2d(
                256,
                4,
                4,
                stride=2,
                padding=1,
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                4,
                256,
                4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                256,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                256,
                128,
                4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                64,
                3,
                4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.proj = nn.Sequential(
            nn.Linear(
                dim,
                dim,
            ),
            nn.SiLU(),
            nn.Linear(
                dim,
                dim,
            ),
        )

    def forward(self, t):
        half = self.dim // 2

        freq = torch.exp(
            torch.arange(
                half,
                device=t.device,
            )
            * (
                -math.log(10000.0)
                / max(half - 1, 1)
            )
        )

        angles = (
            t[:, None]
            * freq[None, :]
        )

        emb = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        return self.proj(emb)


class DiffusionUNet(nn.Module):
    """
    Compact latent-space diffusion network.

    Input:
        4x64x64 latent

    Conditioning:
        text embedding + timestep
    """

    def __init__(
        self,
        vocab_size,
        base=128,
    ):
        super().__init__()

        self.text = TextEncoder(
            vocab_size,
            embed_dim=256,
        )

        self.time = TimeEmbedding(
            256
        )

        self.input = nn.Conv2d(
            4,
            base,
            3,
            padding=1,
        )

        self.block1 = ResBlock(
            base
        )

        self.down = nn.Conv2d(
            base,
            base * 2,
            4,
            stride=2,
            padding=1,
        )

        self.block2 = ResBlock(
            base * 2
        )

        self.mid = ResBlock(
            base * 2
        )

        self.up = nn.ConvTranspose2d(
            base * 2,
            base,
            4,
            stride=2,
            padding=1,
        )

        self.block3 = ResBlock(
            base
        )

        self.output = nn.Conv2d(
            base,
            4,
            3,
            padding=1,
        )

        self.condition = nn.Linear(
            512,
            base * 2,
        )

    def forward(
        self,
        x,
        tokens,
        timestep,
    ):
        text = self.text(tokens)

        time = self.time(timestep)

        conditioning = torch.cat(
            [
                text,
                time,
            ],
            dim=1,
        )

        c = self.condition(
            conditioning
        )

        c = c[:, :, None, None]

        h1 = self.input(x)

        h1 = self.block1(h1)

        h2 = self.down(h1)

        h2 = h2 + c

        h2 = self.block2(h2)

        h2 = self.mid(h2)

        h = self.up(h2)

        h = h + h1

        h = self.block3(h)

        return self.output(h)


class CrystalImageV2(nn.Module):
    def __init__(
        self,
        vocab_size,
    ):
        super().__init__()

        self.autoencoder = AutoEncoder()

        self.diffusion = DiffusionUNet(
            vocab_size
        )
