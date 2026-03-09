"""Script to download the Telco Customer Churn dataset from Kaggle."""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from utils.data_download import download_kaggle_dataset


if __name__ == "__main__":
    download_kaggle_dataset(data_dir=PROJECT_ROOT / "data")
