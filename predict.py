import argparse
from pathlib import Path

import torch

from models.patchtst.predictor import PatchTSTPredictor


def main():
    parser = argparse.ArgumentParser(description="Load a trained PatchTST model and run a minimal prediction.")
    parser.add_argument("--checkpoint", default="outputs/models/patchtst.pt", help="Path to the model checkpoint")
    parser.add_argument("--context-length", type=int, default=96, help="Context length")
    parser.add_argument("--prediction-length", type=int, default=24, help="Prediction length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
        device="cpu",
        checkpoint_path=args.checkpoint,
    )

    context = torch.zeros(args.batch_size, args.context_length)
    predictions = predictor.predict(context)

    output_path = Path("outputs") / "predictions.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(predictions.tolist()))

    print(f"Saved dummy predictions to {output_path}")


if __name__ == "__main__":
    main()
