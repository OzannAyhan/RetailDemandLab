import torch

from data.synthetic import SyntheticRetailDataset
from data.m5_loader import M5Dataset
from data.windows import WindowDataset
from models.patchtst.trainer import PatchTSTTrainer


# -------------------------------------------------------
# Dataset selection
# -------------------------------------------------------

USE_M5 = True


def main():

    if USE_M5:

        print("Loading M5 dataset...")

        generator = M5Dataset(
            sales_path="data/raw/sales_train_evaluation.csv",
            max_series=500,
        )

    else:

        print("Generating synthetic dataset...")

        generator = SyntheticRetailDataset(
            num_series=500,
            series_length=730,
            seasonality=7,
            seed=42,
        )

    panel = generator.generate()

    print(f"Loaded {len(panel)} time series.")

    context_length = 96
    prediction_length = 24

    dataset = WindowDataset(
        panel=panel,
        context_length=context_length,
        horizon=prediction_length,
    )

    print(f"Generated {len(dataset)} training windows.")

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

    model = trainer.fit(dataset)

    torch.save(
        model.state_dict(),
        "outputs/models/patchtst.pt",
    )

    print("\nModel saved to outputs/models/patchtst.pt")


if __name__ == "__main__":
    main()