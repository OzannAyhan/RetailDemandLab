"""
Training pipeline for the PatchTST model.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.windows import WindowDataset
from models.patchtst.network import PatchTSTNetwork
from models.patchtst.loss import PinballLoss


class PatchTSTTrainer:
    """
    Trainer class for the PatchTST forecasting model.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        quantiles: list[float],
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 20,
        device: str = "auto",
        seed: int = 42,
    ) -> None:

        self.context_length = context_length
        self.prediction_length = prediction_length

        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        self.device = self._resolve_device(device)

        torch.manual_seed(seed)

        self.model = PatchTSTNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            n_quantiles=len(quantiles),
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        self.loss_fn = PinballLoss(quantiles)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Automatically select the available compute device.
        """

        if device != "auto":
            return device

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize(
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RevIN-style instance normalization.
        """

        mean = batch.mean(dim=1, keepdim=True)
        std = batch.std(dim=1, keepdim=True) + 1e-5

        normalized = (batch - mean) / std

        return normalized, mean, std

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        """
        Train the model for one epoch.
        """

        self.model.train()

        running_loss = 0.0

        for context, target in dataloader:

            context = context.to(self.device)
            target = target.to(self.device)

            context, mean, std = self._normalize(context)

            predictions = self.model(context)

            predictions = (
                predictions * std.unsqueeze(-1)
                + mean.unsqueeze(-1)
            )

            loss = self.loss_fn(predictions, target)

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def fit(
        self,
        dataset: WindowDataset,
    ) -> PatchTSTNetwork:
        """
        Train the PatchTST model.
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset contains no training windows."
            )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        print(
            f"\nTraining PatchTST "
            f"({len(dataset)} windows)"
        )

        for epoch in range(self.epochs):

            loss = self.train_epoch(dataloader)

            print(
                f"Epoch "
                f"{epoch + 1:02d}/{self.epochs} "
                f"| Pinball Loss: {loss:.4f}"
            )

        return self.model