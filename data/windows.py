"""
Sliding window dataset.

Converts time series into (context, target) training samples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Generates sliding windows for supervised forecasting.

    Each sample contains:

        context ---> model input

        target  ---> expected forecast
    """

    def __init__(
        self,
        panel,
        context_length: int,
        horizon: int,
    ):

        self.samples = []

        total_window = context_length + horizon

        for series in panel:

            values = np.log1p(
                np.clip(series.values, 0, None)
            )

            if len(values) < total_window:
                continue

            stride = max(1, horizon // 2)

            for start in range(
                0,
                len(values) - total_window + 1,
                stride,
            ):

                context = values[
                    start : start + context_length
                ]

                target = values[
                    start + context_length :
                    start + total_window
                ]

                self.samples.append(
                    (context, target)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        context, target = self.samples[index]

        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )