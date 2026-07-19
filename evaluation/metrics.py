"""Forecast evaluation metrics for quantile-based predictions (src copy)."""

from __future__ import annotations

import torch


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return torch.mean(torch.abs(y_true - y_pred))


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return torch.mean(torch.square(y_true - y_pred))


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mean_squared_error(y_true, y_pred))


def normalized_root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    target_range = torch.ptp(y_true)
    if target_range <= epsilon:
        raise ValueError("Cannot compute NRMSE for a constant target tensor")
    return root_mean_squared_error(y_true, y_pred) / target_range


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor | list[float]) -> torch.Tensor:
    if y_pred.dim() != 3:
        raise ValueError("y_pred must have shape (batch, horizon, n_quantiles)")
    if y_true.dim() != 2:
        raise ValueError("y_true must have shape (batch, horizon)")
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_pred.shape[2] != len(quantiles):
        raise ValueError(f"Expected {len(quantiles)} quantiles, got {y_pred.shape[2]}")
    q = torch.as_tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device).view(1, 1, -1)
    error = y_true.unsqueeze(-1) - y_pred
    loss = torch.maximum(q * error, (q - 1) * error)
    return torch.mean(loss)


def weighted_quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor | list[float]) -> torch.Tensor:
    return pinball_loss(y_true, y_pred, quantiles)
