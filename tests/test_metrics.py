import torch

from evaluation.metrics import weighted_absolute_percentage_error


def test_weighted_absolute_percentage_error_handles_basic_case():
    y_true = torch.tensor([10.0, 20.0, 30.0])
    y_pred = torch.tensor([8.0, 24.0, 30.0])

    result = weighted_absolute_percentage_error(y_true, y_pred)

    assert torch.isclose(result, torch.tensor(10.0))


def test_weighted_absolute_percentage_error_includes_zero_targets():
    y_true = torch.tensor([0.0, 10.0])
    y_pred = torch.tensor([1.0, 8.0])

    result = weighted_absolute_percentage_error(y_true, y_pred)

    assert torch.isclose(result, torch.tensor(30.0))
