from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train import summarize_classification


def test_classification_summary_builds_confusion_and_macro_f1() -> None:
    predictions = torch.tensor([0, 1, 1, 3, 2, 0], dtype=torch.long)
    targets = torch.tensor([0, 1, 2, 3, 2, 0], dtype=torch.long)
    summary = summarize_classification(
        predictions,
        targets,
        loss=0.25,
        class_names=["NM", "FM", "FiM", "AFM"],
    )

    assert summary.metrics.accuracy == pytest.approx(5 / 6)
    assert summary.confusion_matrix[2][1] == 1
    assert summary.per_class_metrics["FiM"]["support"] == 2
    assert summary.metrics.macro_f1 > 0.7
