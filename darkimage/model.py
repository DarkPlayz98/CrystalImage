import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                channels,
            ),
        )

    def forward(self, x):
        return torch.relu(
            x + self.block(x)
        )


class CrystalImage(nn.Module):
    """
    Crystal Image v1.2.

    Custom randomly initialized text-conditioned
    denoising network.

    Training resolution:
        96x96

    Final output:
        1536x1536
    """

    def __init__(
        self,
        vocab_size,
        text_dim=64,
        channels=64,
    ):
        super().__init__()

        self.text_dim = text_dim

        self.embedding = nn.Embedding(
            vocab_size,
            text_dim,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_dim,
                channels,
            ),
            nn.SiLU(),
            nn.Linear(
                channels,
                channels,
            ),
        )

        self.time_projection = nn.Sequential(
            nn.Linear(
                1,
                channels,
            ),
            nn.SiLU(),
            nn.Linear(
                channels,
                channels,
            ),
        )

        self.input = nn.Conv2d(
            3,
            channels,
            3,
            padding=1,
        )

        self.res1 = ResidualBlock(
            channels
        )

        self.res2 = ResidualBlock(
            channels
        )

        self.res3 = ResidualBlock(
            channels
        )

        self.output = nn.Conv2d(
            channels,
            3,
            3,
            padding=1,
        )

    def encode_text(
        self,
        token_ids,
    ):
        x = self.embedding(
            token_ids
        )

        return x.mean(
            dim=1
        )

    def forward(
        self,
        noisy,
        token_ids,
        timestep,
    ):
        text = self.encode_text(
            token_ids
        )

        text = self.text_projection(
            text
        )

        time = timestep.reshape(
            -1,
            1,
        )

        time = self.time_projection(
            time
        )

        condition = (
            text + time
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        x = self.input(
            noisy
        )

        x = x + condition

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        return self.output(x)
