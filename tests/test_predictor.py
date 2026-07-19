import torch

from models.patchtst.predictor import PatchTSTPredictor


def test_predictor_outputs_expected_shape_and_loads_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "patchtst_test.pt"

    predictor = PatchTSTPredictor(
        context_length=64,
        prediction_length=8,
        quantiles=[0.1, 0.5, 0.9],
        patch_length=16,
        stride=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        device="cpu",
    )

    context = torch.randn(2, 64)
    predictions = predictor.predict(context)

    assert predictions.shape == (2, 8, 3)

    torch.save(predictor.model.state_dict(), checkpoint_path)

    loaded_predictor = PatchTSTPredictor(
        context_length=64,
        prediction_length=8,
        quantiles=[0.1, 0.5, 0.9],
        patch_length=16,
        stride=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        device="cpu",
        checkpoint_path=str(checkpoint_path),
    )

    loaded_predictions = loaded_predictor.predict(context)

    assert loaded_predictions.shape == (2, 8, 3)
    assert torch.allclose(loaded_predictions, predictions)
