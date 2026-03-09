from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.evaluation import (
    evaluate_model,
    find_best_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
)


class DummyModel:
    def __init__(self, preds: np.ndarray, importances: np.ndarray | None = None) -> None:
        self._preds = preds
        self.feature_importances_ = importances if importances is not None else np.array([0.5, 0.5])

    def predict(self, X):
        return self._preds


def test_evaluate_model_returns_metrics() -> None:
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.3, 0.4])

    model = DummyModel(preds=y_pred)
    metrics = evaluate_model(model, X_test=None, y_test=y_test, y_pred_proba=y_pred_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["roc_auc"] is not None
    assert "threshold" in metrics


def test_find_best_threshold_returns_valid_value() -> None:
    y_test = np.array([0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.8, 0.3, 0.9, 0.6, 0.2])

    threshold, summary = find_best_threshold(y_test, y_pred_proba)

    assert 0.0 <= threshold <= 1.0
    assert set(summary.keys()) == {"precision", "recall", "f1"}


def test_find_best_threshold_with_min_precision() -> None:
    y_test = np.array([0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.8, 0.3, 0.9, 0.6, 0.2])

    _, summary = find_best_threshold(y_test, y_pred_proba, min_precision=0.7)

    assert summary["precision"] >= 0.7


def test_plot_feature_importance_writes_file(tmp_path: Path) -> None:
    model = DummyModel(preds=np.array([0, 1]), importances=np.array([0.2, 0.8, 0.5]))
    feature_names = ["f1", "f2", "f3"]

    plot_feature_importance(model, feature_names, output_dir=tmp_path, top_n=3)

    assert (tmp_path / "feature_importance.png").exists()


def test_plot_confusion_matrix_writes_file(tmp_path: Path) -> None:
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    plot_confusion_matrix(y_test, y_pred, output_dir=tmp_path)

    assert (tmp_path / "confusion_matrix.png").exists()
