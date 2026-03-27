from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train import MultitaskUncertainty


def test_multitask_uncertainty_matches_plain_weight_at_zero_log_var() -> None:
    module = MultitaskUncertainty(["energy"], init_log_var=0.0)
    raw_loss = torch.tensor(2.0, dtype=torch.float32)
    weighted = module.weighted_loss("energy", raw_loss, base_weight=1.5)
    assert torch.isclose(weighted, torch.tensor(3.0))


def test_multitask_uncertainty_backpropagates_log_var() -> None:
    module = MultitaskUncertainty(["ordering"], init_log_var=0.0)
    raw_loss = torch.tensor(1.25, dtype=torch.float32)
    weighted = module.weighted_loss("ordering", raw_loss, base_weight=1.0)
    weighted.backward()
    grad = module.log_vars["ordering"].grad
    assert grad is not None
    assert torch.isfinite(grad)
