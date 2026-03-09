"""Tests for visualization functions."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.evaluation import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_threshold_vs_metrics,
    plot_learning_curves,
    plot_calibration_curve,
    plot_feature_correlation_heatmap,
    plot_churn_rate_by_feature,
)
from models.xgboost_model import build_xgboost_model


class TestVisualizations:
    """Test visualization functions."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        y_test = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        return y_test, y_pred_proba

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for churn rate plots."""
        return pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'] * 30 + ['Month-to-month'] * 10,
            'Churn': ['Yes', 'No'] * 50,
        })

    def test_plot_roc_curve_creates_file(self, sample_predictions, tmp_path):
        """Test that ROC curve plot creates output file."""
        y_test, y_pred_proba = sample_predictions
        
        plot_roc_curve(y_test, y_pred_proba, tmp_path)
        
        assert (tmp_path / "roc_curve.png").exists()

    def test_plot_precision_recall_curve_creates_file(self, sample_predictions, tmp_path):
        """Test that Precision-Recall curve creates output file."""
        y_test, y_pred_proba = sample_predictions
        
        plot_precision_recall_curve(y_test, y_pred_proba, tmp_path, selected_threshold=0.5)
        
        assert (tmp_path / "precision_recall_curve.png").exists()

    def test_plot_precision_recall_curve_without_threshold(self, sample_predictions, tmp_path):
        """Test Precision-Recall curve without marking threshold."""
        y_test, y_pred_proba = sample_predictions
        
        plot_precision_recall_curve(y_test, y_pred_proba, tmp_path)
        
        assert (tmp_path / "precision_recall_curve.png").exists()

    def test_plot_class_distribution_creates_file(self, tmp_path):
        """Test that class distribution plot creates output file."""
        y_before = pd.Series([0] * 80 + [1] * 20)
        y_after = pd.Series([0] * 80 + [1] * 80)
        
        plot_class_distribution(y_before, y_after, tmp_path)
        
        assert (tmp_path / "class_distribution_smote.png").exists()

    def test_plot_threshold_vs_metrics_creates_file(self, sample_predictions, tmp_path):
        """Test that threshold vs metrics plot creates output file."""
        y_test, y_pred_proba = sample_predictions
        
        plot_threshold_vs_metrics(y_test, y_pred_proba, tmp_path, selected_threshold=0.5)
        
        assert (tmp_path / "threshold_vs_metrics.png").exists()

    def test_plot_learning_curves_creates_file(self, sample_training_data, tmp_path):
        """Test that learning curves plot creates output file."""
        X, y = sample_training_data
        # Build model without early stopping for testing
        model = build_xgboost_model(early_stopping_rounds=None)
        model.fit(X, y)
        
        plot_learning_curves(model, X, y, tmp_path, cv=2)
        
        assert (tmp_path / "learning_curves.png").exists()

    def test_plot_calibration_curve_creates_file(self, sample_predictions, tmp_path):
        """Test that calibration curve creates output file."""
        y_test, y_pred_proba = sample_predictions
        
        plot_calibration_curve(y_test, y_pred_proba, tmp_path, n_bins=5)
        
        assert (tmp_path / "calibration_curve.png").exists()

    def test_plot_feature_correlation_heatmap_creates_file(self, sample_training_data, tmp_path):
        """Test that feature correlation heatmap creates output file."""
        X, _ = sample_training_data
        
        plot_feature_correlation_heatmap(X, tmp_path, top_n=5)
        
        assert (tmp_path / "feature_correlation_heatmap.png").exists()

    def test_plot_churn_rate_by_feature_creates_file(self, sample_dataframe, tmp_path):
        """Test that churn rate by feature plot creates output file."""
        plot_churn_rate_by_feature(sample_dataframe, 'Contract', tmp_path)
        
        assert (tmp_path / "churn_rate_by_contract.png").exists()

    def test_plot_churn_rate_handles_missing_column(self, sample_dataframe, tmp_path):
        """Test that churn rate plot handles missing column gracefully."""
        # Should not raise error
        plot_churn_rate_by_feature(sample_dataframe, 'NonExistentColumn', tmp_path)
        
        # Should not create file
        assert not (tmp_path / "churn_rate_by_nonexistentcolumn.png").exists()

    def test_plot_churn_rate_handles_missing_churn_column(self, tmp_path):
        """Test that churn rate plot handles missing Churn column."""
        df = pd.DataFrame({'Contract': ['A', 'B', 'C']})
        
        # Should not raise error
        plot_churn_rate_by_feature(df, 'Contract', tmp_path)

    def test_all_plots_create_output_dir(self, sample_predictions, sample_training_data, tmp_path):
        """Test that all plotting functions create output directory if it doesn't exist."""
        y_test, y_pred_proba = sample_predictions
        X, y = sample_training_data
        
        nested_dir = tmp_path / "nested" / "output"
        
        # Should create directory and file
        plot_roc_curve(y_test, y_pred_proba, nested_dir)
        
        assert nested_dir.exists()
        assert (nested_dir / "roc_curve.png").exists()
