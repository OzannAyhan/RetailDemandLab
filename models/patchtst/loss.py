"""
Loss functions for probabilistic forecasting.
"""

from typing import Sequence

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss.

    Parameters
    ----------
    quantile_levels : Sequence[float]
        Quantiles predicted by the model (e.g. [0.1, 0.5, 0.9]).
    """

    def __init__(self, quantile_levels: Sequence[float]):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the average pinball loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Shape: (batch_size, prediction_length, num_quantiles)

        targets : torch.Tensor
            Shape: (batch_size, prediction_length)

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        errors = targets.unsqueeze(-1) - predictions

        loss = torch.maximum(
            self.quantiles.view(1, 1, -1) * errors,
            (self.quantiles.view(1, 1, -1) - 1.0) * errors,
        )

        return loss.mean()