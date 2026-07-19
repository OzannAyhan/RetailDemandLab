import json
from pathlib import Path

import torch

from data.synthetic import SyntheticRetailDataset
from data.windows import WindowDataset
from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    pinball_loss,
    weighted_quantile_loss,
)
from models.patchtst.predictor import PatchTSTPredictor


if __name__ == "__main__":
    generator = SyntheticRetailDataset(num_series=20, series_length=220, seasonality=7, seed=42)
    panel = generator.generate()

    dataset = WindowDataset(panel=panel, context_length=96, horizon=24)

    if len(dataset) == 0:
        raise RuntimeError("No windows were generated for evaluation")

    context, target = dataset[0]
    quantiles = [0.1, 0.5, 0.9]

    predictor = PatchTSTPredictor(
        context_length=96,
        prediction_length=24,
        quantiles=quantiles,
        patch_length=32,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        device="cpu",
        checkpoint_path="outputs/models/patchtst.pt",
    )

    predictions = predictor.predict(context.unsqueeze(0))
    predictions = predictions.squeeze(0)
    target = target

    print("Predictions shape:", predictions.shape)
    print("Predictions (first 3 steps):")
    print(predictions[:3])

    median_predictions = predictions[:, 1]
    mae = mean_absolute_error(target, median_predictions)
    rmse = root_mean_squared_error(target, median_predictions)
    pinball = pinball_loss(target.unsqueeze(0), predictions.unsqueeze(0), quantiles)
    wql = weighted_quantile_loss(target.unsqueeze(0), predictions.unsqueeze(0), quantiles)

    metrics = {
        "mae": round(mae.item(), 6),
        "rmse": round(rmse.item(), 6),
        "pinball_loss": round(pinball.item(), 6),
        "weighted_quantile_loss": round(wql.item(), 6),
    }

    print("\nSummary:")
    for name, value in metrics.items():
        print(f"- {name}: {value}")

    output_path = Path("outputs/metrics_summary.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {output_path}")
