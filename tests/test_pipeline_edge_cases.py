from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

import models.xgboost_model as xgb_pipeline
from utils.evaluation import evaluate_model


class ConstantPredModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_evaluate_model_without_proba_sets_roc_auc_none() -> None:
    model = ConstantPredModel()
    y_test = np.array([0, 1, 0, 1])

    metrics = evaluate_model(model, X_test=np.zeros((4, 2)), y_test=y_test, y_pred_proba=None)

    assert metrics["roc_auc"] is None


def test_training_pipeline_missing_dataset_returns_cleanly(tmp_path: Path) -> None:
    data_path = tmp_path / "missing.csv"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"

    xgb_pipeline.run_training_pipeline(
        data_path=data_path,
        models_dir=models_dir,
        output_dir=output_dir,
    )

    assert models_dir.exists()
    assert output_dir.exists()
    assert not (models_dir / "xgboost_churn_model.pkl").exists()
    assert not (models_dir / "scaler.pkl").exists()
