"""
Synthetic retail demand generator.

Creates realistic retail-like time series for testing and development
without requiring the M5 dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticRetailDataset:
    """
    Generate synthetic retail demand time series.

    The generated series mimic common retail demand patterns:

    - Dense
    - Erratic
    - Intermittent
    - Lumpy

    The output is a list of pandas Series compatible with
    WindowDataset.
    """

    def __init__(
        self,
        num_series: int = 500,
        series_length: int = 730,
        seasonality: int = 7,
        seed: int = 42,
    ) -> None:
        self.num_series = num_series
        self.series_length = series_length
        self.seasonality = seasonality
        self.rng = np.random.default_rng(seed)

        self.profiles = (
            "dense",
            "erratic",
            "intermittent",
            "lumpy",
        )

        self.profile_weights = np.array(
            [0.30, 0.20, 0.30, 0.20]
        )

    def generate(self) -> list[pd.Series]:
        """
        Generate a panel of synthetic demand series.
        """

        panel: list[pd.Series] = []

        for _ in range(self.num_series):

            profile = self.rng.choice(
                self.profiles,
                p=self.profile_weights,
            )

            values = self._generate_profile(profile)

            panel.append(pd.Series(values))

        return panel

    def _generate_profile(
        self,
        profile: str,
    ) -> np.ndarray:

        if profile == "dense":
            return self._generate_dense()

        if profile == "erratic":
            return self._generate_erratic()

        if profile == "intermittent":
            return self._generate_intermittent()

        return self._generate_lumpy()

    def _weekly_seasonality(
        self,
        amplitude: float,
    ) -> np.ndarray:

        phase = self.rng.uniform(
            0,
            2 * np.pi,
        )

        t = np.arange(self.series_length)

        return amplitude * (
            1.0
            + 0.5
            * np.sin(
                2 * np.pi * t / self.seasonality
                + phase
            )
        )

    def _generate_dense(self) -> np.ndarray:

        base = self.rng.uniform(8, 40)

        seasonality = self._weekly_seasonality(base)

        trend = np.linspace(
            0,
            self.rng.uniform(-0.2, 0.4) * base,
            self.series_length,
        )

        lam = np.clip(
            seasonality + trend,
            0.1,
            None,
        )

        return self.rng.poisson(lam).astype(np.float64)

    def _generate_erratic(self) -> np.ndarray:

        base = self.rng.uniform(3, 12)

        seasonality = self._weekly_seasonality(base)

        noise = self.rng.gamma(
            shape=1.2,
            scale=base / 2,
            size=self.series_length,
        )

        demand = np.clip(
            seasonality * 0.4 + noise,
            0,
            None,
        )

        return np.round(demand).astype(np.float64)

    def _generate_intermittent(self) -> np.ndarray:

        probability = self.rng.uniform(
            0.08,
            0.25,
        )

        occurrence = (
            self.rng.random(self.series_length)
            < probability
        )

        sizes = (
            self.rng.poisson(
                self.rng.uniform(1.0, 3.0),
                size=self.series_length,
            )
            + 1
        )

        return (
            occurrence * sizes
        ).astype(np.float64)

    def _generate_lumpy(self) -> np.ndarray:

        probability = self.rng.uniform(
            0.05,
            0.15,
        )

        occurrence = (
            self.rng.random(self.series_length)
            < probability
        )

        sizes = self.rng.gamma(
            shape=2.0,
            scale=self.rng.uniform(4, 10),
            size=self.series_length,
        )

        return np.round(
            occurrence * sizes
        ).astype(np.float64)
