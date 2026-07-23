"""
M5 dataset loader.
"""

from __future__ import annotations

import pandas as pd


class M5Dataset:
    """
    Loads the M5 Walmart sales dataset.
    """

    def __init__(
        self,
        sales_path: str,
        max_series: int | None = None,
    ) -> None:

        self.sales_path = sales_path
        self.max_series = max_series

    def generate(self) -> list[pd.Series]:
        """
        Returns a list of pandas Series.
        """

        sales = pd.read_csv(self.sales_path)

        if self.max_series is not None:
            sales = sales.head(self.max_series)

        demand_columns = [
            c
            for c in sales.columns
            if c.startswith("d_")
        ]

        demand = sales[demand_columns].to_numpy(dtype=float)

        ids = sales["id"].tolist()

        panel = [
            pd.Series(
                demand[i],
                name=ids[i],
            )
            for i in range(len(ids))
        ]

        return panel