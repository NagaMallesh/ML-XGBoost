from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

import models.xgboost_model as xgb_pipeline


class DummyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_count: int = 10) -> None:
        self.feature_count = feature_count
        self.feature_importances_ = np.ones(feature_count)

    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility."""
        self.feature_importances_ = np.ones(X.shape[1])
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([i % 2 for i in range(len(X))], dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2), dtype=float)
        probs[:, 1] = 0.5
        return probs
        return probs


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004"],
            "gender": ["Male", "Female", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "PhoneService": ["Yes", "No", "Yes", "No"],
            "MultipleLines": ["No", "No phone service", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["Yes", "No", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "Yes"],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["29.85", "1889.50", "108.15", ""],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


def test_run_training_pipeline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "telco.csv"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"

    df = _sample_dataframe()
    df.to_csv(data_path, index=False)

    def _train_stub(X_train, y_train, X_test, y_test, random_state=42, **kwargs):
        return DummyModel(feature_count=X_train.shape[1])

    monkeypatch.setattr(xgb_pipeline, "train_xgboost_model", _train_stub)

    xgb_pipeline.run_training_pipeline(
        data_path=data_path,
        models_dir=models_dir,
        output_dir=output_dir,
        test_size=0.5,
        random_state=42,
    )

    assert (models_dir / "xgboost_churn_model.pkl").exists()
    assert (models_dir / "scaler.pkl").exists()
    assert (output_dir / "feature_importance.png").exists()
    assert (output_dir / "confusion_matrix.png").exists()
