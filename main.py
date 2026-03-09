"""Menu-driven CLI for ML-XGBoost."""

from pathlib import Path
import subprocess
import sys
import platform

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.xgboost_model import run_training_pipeline
from utils.data_download import download_kaggle_dataset


def print_menu() -> None:
    print("\n" + "=" * 50)
    print("ML-XGBoost Menu")
    print("=" * 50)
    print("1. Download dataset")
    print("2. Train XGBoost model")
    print("3. View generated visualizations")
    print("4. Run unit tests")
    print("5. Exit")


def run_unit_tests() -> int:
    """Run unit tests using pytest and return the process exit code."""
    print("\nRunning tests...")
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


def view_visualizations() -> None:
    """Display all generated visualization files in a single matplotlib window."""
    output_dir = PROJECT_ROOT / "output"
    
    if not output_dir.exists():
        print(f"\n✗ Output directory not found: {output_dir}")
        print("  Please train the model first (option 2) to generate visualizations.")
        return
    
    # Find all PNG files in output directory
    image_files = sorted(output_dir.glob("*.png"))
    
    if not image_files:
        print(f"\n✗ No visualization files found in {output_dir}")
        print("  Please train the model first (option 2) to generate visualizations.")
        return
    
    print(f"\n📊 Found {len(image_files)} visualization(s):")
    for img in image_files:
        print(f"   • {img.name}")
    
    print("\n🖼️  Opening visualizations in matplotlib viewer...")
    print("    (Close the window to return to the menu)\n")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        # Calculate grid dimensions (prefer wider layouts)
        n_images = len(image_files)
        n_cols = min(3, n_images)  # Max 3 columns
        n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle('ML-XGBoost Visualizations', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier iteration
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Display each image
        for idx, img_path in enumerate(image_files):
            img = mpimg.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(img_path.stem.replace('_', ' ').title(), 
                               fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        print("✓ Visualization window opened")
        print("  Tip: Use the toolbar to zoom, pan, or save individual plots")
        plt.show()
        print("\n✓ Visualization window closed")
        
    except ImportError as e:
        print(f"✗ Error: matplotlib not available: {e}")
        print(f"  Falling back to system viewer...")
        # Fallback to system viewer
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", str(output_dir)], check=True)
                print(f"✓ Opened output directory: {output_dir}")
        except Exception as fallback_error:
            print(f"✗ Error: {fallback_error}")
            print(f"  Please manually open files in: {output_dir}")
    except Exception as e:
        print(f"✗ Error displaying visualizations: {e}")
        print(f"  Please manually open files in: {output_dir}")


def main() -> None:
    while True:
        print_menu()
        choice = input("\nSelect an option (1-5): ").strip()

        if choice == "1":
            download_kaggle_dataset()
        elif choice == "2":
            data_path = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            models_dir = PROJECT_ROOT / "models"
            output_dir = PROJECT_ROOT / "output"
            run_training_pipeline(
                data_path=data_path,
                models_dir=models_dir,
                output_dir=output_dir,
            )
        elif choice == "3":
            view_visualizations()
        elif choice == "4":
            run_unit_tests()
        elif choice == "5":
            print("Exiting. Bye!")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
