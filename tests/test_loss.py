import torch

from models.patchtst.loss import PinballLoss


def main():
    loss_fn = PinballLoss([0.1, 0.5, 0.9])

    predictions = torch.randn(8, 24, 3)
    targets = torch.randn(8, 24)

    loss = loss_fn(predictions, targets)

    print(f"Loss: {loss.item():.4f}")
    print("✅ PinballLoss test passed.")


if __name__ == "__main__":
    main()