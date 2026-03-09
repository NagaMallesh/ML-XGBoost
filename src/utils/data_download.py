"""Dataset download utilities."""

from pathlib import Path


def download_kaggle_dataset(data_dir: Path | None = None) -> bool:
    """Download and extract the Telco Customer Churn dataset from Kaggle."""
    target_dir = data_dir or (Path(__file__).resolve().parents[2] / "data")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")

    try:
        import kaggle

        kaggle.api.dataset_download_files(
            "blastchar/telco-customer-churn",
            path=str(target_dir),
            unzip=True,
        )

        print(f"✓ Dataset downloaded successfully to {target_dir}")

        files = list(target_dir.glob("*.csv"))
        if files:
            print("\nDownloaded files:")
            for file in files:
                print(f"  - {file.name}")

    except Exception as exc:
        print(f"✗ Error downloading dataset: {exc}")
        print("\nMake sure you have:")
        print("1. Created a Kaggle account")
        print("2. Placed your kaggle.json in ~/.kaggle/")
        print("3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False

    return True
