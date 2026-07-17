"""
PatchTST Network

Defines only the neural network architecture.
Training and inference are handled elsewhere.
"""

import torch
from torch import nn


class PatchTSTNetwork(nn.Module):
    def __init__(
        self,
        context: int,
        horizon: int,
        n_quantiles: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.context = context
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = (context - patch_len) // stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Learnable positional embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Forecast head
        self.head = nn.Linear(
            d_model * self.n_patches,
            horizon * n_quantiles,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a sequence into overlapping patches.

        Input:
            (batch, context)

        Output:
            (batch, patches, patch_length)
        """
        return x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = self.patchify(x)

        embeddings = self.patch_embedding(patches)
        embeddings = embeddings + self.position_embedding

        encoded = self.encoder(embeddings)

        encoded = encoded.flatten(start_dim=1)

        output = self.head(encoded)

        return output.view(
            -1,
            self.horizon,
            self.n_quantiles,
        )