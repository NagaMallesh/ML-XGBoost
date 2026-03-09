"""Argparse CLI for ML-XGBoost."""

from pathlib import Path
import argparse
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.xgboost_model import run_training_pipeline
from utils.data_download import download_kaggle_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML-XGBoost CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download Kaggle dataset")
    download_parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory to store the dataset",
    )

    train_parser = subparsers.add_parser("train", help="Train the XGBoost model")
    train_parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the input CSV dataset",
    )
    train_parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save trained models",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory to save plots and artifacts",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable randomized hyperparameter tuning before final training",
    )
    train_parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=15,
        help="Number of randomized search iterations when tuning is enabled",
    )
    train_parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Disable decision-threshold tuning (use default 0.5)",
    )
    train_parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Optional minimum precision constraint during threshold tuning",
    )
    train_parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling of minority class (useful for large imbalanced datasets)",
    )
    train_parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        help="Disable feature engineering (use raw features only)",
    )

    subparsers.add_parser("test", help="Run unit tests")

    return parser


def run_unit_tests() -> int:
    """Run unit tests using pytest and return the process exit code."""
    print("Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode == 0:
        print("✓ Tests passed")
    else:
        print(f"✗ Tests failed with exit code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        download_kaggle_dataset(data_dir=args.data_dir)
    elif args.command == "train":
        run_training_pipeline(
            data_path=args.data_path,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            tune_hyperparameters=args.tune_hyperparameters,
            tuning_iterations=args.tuning_iterations,
            tune_threshold=not args.no_threshold_tuning,
            min_precision=args.min_precision,
            use_smote=not args.no_smote,
            apply_feature_engineering=not args.no_feature_engineering,
        )
    elif args.command == "test":
        run_unit_tests()


if __name__ == "__main__":
    main()
