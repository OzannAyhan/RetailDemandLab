"""
Inference pipeline for the PatchTST forecasting model (src copy).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from models.patchtst.network import PatchTSTNetwork


class PatchTSTPredictor:
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        quantiles: list[float],
        patch_length: int = 32,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.device = self._resolve_device(device)

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

        self.model.eval()

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = batch.mean(dim=1, keepdim=True)
        std = batch.std(dim=1, keepdim=True) + 1e-5
        normalized = (batch - mean) / std
        return normalized, mean, std

    def load(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() != 2:
            raise ValueError("context must have shape (batch, context_length)")

        if context.size(1) != self.context_length:
            raise ValueError(
                f"context length must be {self.context_length}, got {context.size(1)}"
            )

        with torch.no_grad():
            context = context.to(self.device)
            normalized, mean, std = self._normalize(context)
            predictions = self.model(normalized)
            predictions = predictions * std.unsqueeze(-1) + mean.unsqueeze(-1)

        predictions = torch.expm1(predictions)
        predictions = torch.clamp(predictions, min=0.0)

        return predictions
