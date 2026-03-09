"""Tests for SMOTE oversampling functionality."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.xgboost_model import apply_smote


class TestSMOTE:
    """Test SMOTE oversampling module."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create an imbalanced dataset."""
        # 90% negative, 10% positive
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        })
        y = pd.Series([0] * 90 + [1] * 10)
        return X, y

    def test_apply_smote_balances_classes(self, imbalanced_data):
        """Test that SMOTE creates synthetic samples to balance classes."""
        X, y = imbalanced_data
        original_positive_count = (y == 1).sum()
        original_negative_count = (y == 0).sum()
        
        X_smote, y_smote = apply_smote(X, y)
        
        # After SMOTE, minority class should be expanded
        new_positive_count = (y_smote == 1).sum()
        assert new_positive_count > original_positive_count
        
        # Total samples should increase
        assert len(X_smote) > len(X)

    def test_apply_smote_maintains_feature_columns(self, imbalanced_data):
        """Test that SMOTE preserves feature column names."""
        X, y = imbalanced_data
        X_smote, y_smote = apply_smote(X, y)
        
        assert list(X.columns) == list(X_smote.columns)
        assert X_smote.shape[1] == X.shape[1]

    def test_apply_smote_returns_valid_targets(self, imbalanced_data):
        """Test that SMOTE output contains only valid class labels."""
        X, y = imbalanced_data
        X_smote, y_smote = apply_smote(X, y)
        
        assert set(y_smote) == {0, 1}
        assert len(y_smote) == len(X_smote)

    def test_apply_smote_with_balanced_data(self):
        """Test SMOTE on already balanced data."""
        X = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        })
        y = pd.Series([0] * 25 + [1] * 25)
        
        X_smote, y_smote = apply_smote(X, y)
        
        # SMOTE might still create some synthetic samples for consistency
        # but the dataset should remain roughly balanced
        assert (y_smote == 0).sum() > 0
        assert (y_smote == 1).sum() > 0

    def test_apply_smote_reproducibility(self, imbalanced_data):
        """Test that SMOTE with same random_state produces same output."""
        X, y = imbalanced_data
        
        X_smote1, y_smote1 = apply_smote(X, y, random_state=42)
        X_smote2, y_smote2 = apply_smote(X, y, random_state=42)
        
        # With same random state, outputs should be identical
        assert np.allclose(X_smote1.values, X_smote2.values)
        assert (y_smote1 == y_smote2).all()

    def test_apply_smote_increases_minority_class_count(self, imbalanced_data):
        """Test that minority class count increases significantly."""
        X, y = imbalanced_data
        original_minority_count = (y == 1).sum()
        
        X_smote, y_smote = apply_smote(X, y)
        new_minority_count = (y_smote == 1).sum()
        
        # Minority class should at least double or match majority
        assert new_minority_count >= original_minority_count * 2
