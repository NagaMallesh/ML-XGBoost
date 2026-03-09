"""XGBoost model training for Telco Customer Churn prediction."""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from models.xgboost_model import run_training_pipeline


def main() -> None:
    """Main pipeline wrapper for training the XGBoost model."""
    data_path = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    models_dir = PROJECT_ROOT / "models"
    output_dir = PROJECT_ROOT / "output"

    run_training_pipeline(
        data_path=data_path,
        models_dir=models_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
