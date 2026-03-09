"""Tests for feature engineering functions."""

import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.data import engineer_features, preprocess_data, load_data


class TestFeatureEngineering:
    """Test feature engineering module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample telco data for testing."""
        return pd.DataFrame({
            "customerID": ["001", "002", "003"],
            "tenure": [1, 12, 36],
            "MonthlyCharges": [65.0, 85.0, 120.0],
            "TotalCharges": ["65", "1020", "4320"],
            "gender": ["Male", "Female", "Male"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "Yes"],
            "TechSupport": ["No", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "Yes"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Churn": ["No", "Yes", "No"],
        })

    def test_engineer_features_creates_tenure_phases(self, sample_data):
        """Test that tenure phase features are created correctly."""
        result = engineer_features(sample_data)
        
        # Check that tenure phase columns exist
        assert "tenure_phase_early" in result.columns
        assert "tenure_phase_mid" in result.columns
        assert "tenure_phase_late" in result.columns
        
        # Verify values for known tenures
        assert result.iloc[0]["tenure_phase_early"] == 1  # tenure=1 is early
        assert result.iloc[1]["tenure_phase_mid"] == 1    # tenure=12 is mid
        assert result.iloc[2]["tenure_phase_late"] == 1   # tenure=36 is late

    def test_engineer_features_creates_spending_features(self, sample_data):
        """Test that spending-related features are created."""
        result = engineer_features(sample_data)
        
        assert "monthly_to_total_ratio" in result.columns
        assert "is_high_spender" in result.columns
        assert "spending_volatility" in result.columns
        
        # Check that ratios are reasonable (0-1 range)
        assert 0 <= result["monthly_to_total_ratio"].max() <= 1

    def test_engineer_features_handles_zero_total_charges(self):
        """Test handling of zero TotalCharges."""
        data = pd.DataFrame({
            "tenure": [1],
            "MonthlyCharges": [50.0],
            "TotalCharges": ["0"],
            "InternetService": ["DSL"],
        })
        result = engineer_features(data)
        
        # Should not raise error, should handle gracefully
        assert "monthly_to_total_ratio" in result.columns
        assert result.iloc[0]["monthly_to_total_ratio"] == 0

    def test_engineer_features_creates_service_count(self, sample_data):
        """Test that service count feature is created as placeholder."""
        result = engineer_features(sample_data)
        
        assert "service_count" in result.columns
        assert "has_internet" in result.columns

    def test_preprocess_data_with_feature_engineering(self, sample_data):
        """Test preprocessing with feature engineering enabled."""
        X, y = preprocess_data(sample_data, apply_feature_engineering=True)
        
        # Should have more features with engineering
        assert "tenure_phase_early" in X.columns or "months_since_signup" in X.columns
        assert len(X) == 3
        assert len(y) == 3

    def test_preprocess_data_without_feature_engineering(self, sample_data):
        """Test preprocessing without feature engineering."""
        X_with_eng, y_with_eng = preprocess_data(sample_data, apply_feature_engineering=True)
        X_without_eng, y_without_eng = preprocess_data(sample_data, apply_feature_engineering=False)
        
        # Without engineering should have fewer features (no tenure phases, spending features)
        assert len(X_without_eng.columns) < len(X_with_eng.columns)

    def test_engineer_features_preserves_data_integrity(self, sample_data):
        """Test that feature engineering doesn't lose original rows."""
        original_shape = sample_data.shape[0]
        result = engineer_features(sample_data)
        
        assert result.shape[0] == original_shape
        assert all(result["tenure"] == sample_data["tenure"])
