from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.data import load_data, preprocess_data


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
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["29.85", "1889.50", "108.15", ""],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


def test_load_data_reads_csv(tmp_path: Path) -> None:
    df = _sample_dataframe()
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_data(csv_path)

    assert loaded.shape == df.shape
    assert list(loaded.columns) == list(df.columns)


def test_preprocess_data_outputs_expected_columns() -> None:
    df = _sample_dataframe()

    X, y = preprocess_data(df)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert X.isnull().sum().sum() == 0
    assert any(col.startswith("gender_") for col in X.columns)
    assert any(col.startswith("InternetService_") for col in X.columns)
    assert any(col.startswith("Contract_") for col in X.columns)
    assert any(col.startswith("PaymentMethod_") for col in X.columns)
