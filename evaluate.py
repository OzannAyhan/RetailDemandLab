import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.m5_loader import M5Dataset
from evaluation.metrics import mean_absolute_error, root_mean_squared_error, weighted_absolute_percentage_error
from models.patchtst.predictor import PatchTSTPredictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained PatchTST model on the M5 dataset.")
    parser.add_argument("--checkpoint", default="outputs/models/patchtst.pt", help="Path to the model checkpoint")
    parser.add_argument("--sales-path", default="data/raw/sales_train_evaluation.csv", help="Path to the M5 sales CSV")
    parser.add_argument("--context-length", type=int, default=96, help="Context length")
    parser.add_argument("--prediction-length", type=int, default=24, help="Prediction length")
    parser.add_argument("--max-series", type=int, default=500, help="Number of series to evaluate")
    parser.add_argument("--device", default="cpu", help="Computation device")
    args = parser.parse_args()

    predictor = PatchTSTPredictor(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        patch_length=32,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )

    dataset = M5Dataset(
        sales_path=args.sales_path,
        max_series=args.max_series,
    )
    panel = dataset.generate()

    rows = []
    for series in tqdm(panel, desc="Evaluating series"):
        try:
            values = series.to_numpy(dtype=float)
            if len(values) < args.context_length + args.prediction_length:
                raise ValueError("Series is too short")

            history = np.log1p(
                np.clip(
                    values[-(args.context_length + args.prediction_length) : -args.prediction_length],
                    0.0,
                    None,
                )
            )
            target = values[-args.prediction_length:]

            context = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
            predictions = predictor.predict(context)
            if predictions.shape != (1, args.prediction_length, len(predictor.quantiles)):
                raise ValueError(
                    f"Unexpected prediction shape: {predictions.shape}"
                )

            median_forecast = predictions[0, :, 1].cpu().reshape(-1)

            y_true = torch.tensor(target, dtype=torch.float32)
            y_pred = median_forecast.reshape_as(y_true)

            mae = mean_absolute_error(y_true, y_pred).item()
            rmse = root_mean_squared_error(y_true, y_pred).item()
            wape = weighted_absolute_percentage_error(y_true, y_pred).item()

            rows.append(
                {
                    "series_id": series.name,
                    "mae": mae,
                    "rmse": rmse,
                    "wape": wape,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "series_id": series.name,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "wape": float("nan"),
                }
            )
            print(f"Skipping series {series.name}: {exc}")

    results = pd.DataFrame(rows)
    output_path = Path("outputs") / "evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    valid_results = results.dropna(subset=["wape"])
    if valid_results.empty:
        print("No valid series were evaluated.")
        return

    print(f"Number of evaluated series: {len(valid_results)}")
    print(f"Average MAE: {valid_results['mae'].mean():.4f}")
    print(f"Average RMSE: {valid_results['rmse'].mean():.4f}")
    print(f"Average WAPE: {valid_results['wape'].mean():.4f}")

    best_series = valid_results.loc[valid_results["wape"].idxmin()]
    worst_series = valid_results.loc[valid_results["wape"].idxmax()]

    print(f"Best-performing series: {best_series['series_id']} (WAPE={best_series['wape']:.4f})")
    print(f"Worst-performing series: {worst_series['series_id']} (WAPE={worst_series['wape']:.4f})")
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
