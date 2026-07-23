import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.m5_loader import M5Dataset
from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from models.patchtst.predictor import PatchTSTPredictor


def main():
    parser = argparse.ArgumentParser(description="Load a trained PatchTST model and run a real forecast on the M5 dataset.")
    parser.add_argument("--checkpoint", default="outputs/models/patchtst.pt", help="Path to the model checkpoint")
    parser.add_argument("--sales-path", default="data/raw/sales_train_evaluation.csv", help="Path to the M5 sales CSV")
    parser.add_argument("--context-length", type=int, default=96, help="Context length")
    parser.add_argument("--prediction-length", type=int, default=24, help="Prediction length")
    parser.add_argument("--series-index", type=int, default=0, help="Index of the series to evaluate")
    parser.add_argument("--plot-path", default="outputs/forecast.png", help="Path to save the forecast plot")
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

    dataset = M5Dataset(sales_path=args.sales_path, max_series=args.series_index + 1)
    panel = dataset.generate()

    if args.series_index >= len(panel):
        raise IndexError(f"Series index {args.series_index} is out of range for {len(panel)} series")

    series = panel[args.series_index]
    values = np.asarray(series.to_numpy(dtype=float), dtype=np.float32)

    if len(values) < args.context_length + args.prediction_length:
        raise ValueError(
            f"Series {series.name} is too short for context={args.context_length} and horizon={args.prediction_length}"
        )

    window = values[-(args.context_length + args.prediction_length) :]
    context_values = np.log1p(np.clip(window[:-args.prediction_length], 0.0, None))
    target_values = window[-args.prediction_length :]

    context = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
    predictions = predictor.predict(context)

    point_predictions = predictions[0, :, 1].cpu().numpy()
    point_predictions = np.clip(point_predictions, 0.0, None)

    y_true = torch.tensor(target_values, dtype=torch.float32)
    y_pred = torch.tensor(point_predictions, dtype=torch.float32)

    mae = mean_absolute_error(y_true, y_pred).item()
    rmse = root_mean_squared_error(y_true, y_pred).item()
    wape = weighted_absolute_percentage_error(y_true, y_pred).item()

    output_path = Path(args.plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(values)), values, label="Historical demand", color="gray", linewidth=1.5)
    forecast_index = np.arange(len(values), len(values) + args.prediction_length)
    plt.plot(forecast_index, point_predictions, label="Predicted demand", color="tab:blue", linewidth=2)
    plt.plot(forecast_index, target_values, label="Actual demand", color="tab:orange", linewidth=2)
    plt.axvline(len(values) - args.prediction_length, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time step")
    plt.ylabel("Demand")
    plt.title(f"Series {series.name} forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Series ID: {series.name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4f}")
    print(f"Saved forecast plot to {output_path}")


if __name__ == "__main__":
    main()
