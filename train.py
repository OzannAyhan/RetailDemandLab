import torch

from data.synthetic import SyntheticRetailDataset
from data.windows import WindowDataset
from models.patchtst.trainer import PatchTSTTrainer


def main():

    # ------------------------------------------------------------------
    # Generate synthetic retail data
    # ------------------------------------------------------------------

    generator = SyntheticRetailDataset(
        num_series=500,
        series_length=730,
        seasonality=7,
        seed=42,
    )

    panel = generator.generate()

    # ------------------------------------------------------------------
    # Create sliding-window dataset
    # ------------------------------------------------------------------

    context_length = 96
    prediction_length = 24

    dataset = WindowDataset(
        panel=panel,
        context_length=context_length,
        horizon=prediction_length,
    )

    print(f"Generated {len(dataset)} training windows.")

    # ------------------------------------------------------------------
    # Configure PatchTST trainer
    # ------------------------------------------------------------------

    trainer = PatchTSTTrainer(
        context_length=context_length,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        patch_length=32,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=64,
        epochs=5,
        device="auto",
        seed=42,
    )

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------

    model = trainer.fit(dataset)

    # ------------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------------

    torch.save(
        model.state_dict(),
        "outputs/models/patchtst.pt",
    )

    print("\nModel saved to outputs/models/patchtst.pt")


if __name__ == "__main__":
    main()