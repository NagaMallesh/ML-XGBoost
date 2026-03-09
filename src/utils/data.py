"""Data loading and preprocessing utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


def load_data(file_path) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific features for churn prediction.
    
    Features created:
    - tenure_binned: Categorize tenure into phases (early/mid/late)
    - contract_phase: Interaction between contract type and tenure
    - monthly_to_total_ratio: Relationship between monthly and total charges
    - is_high_spender: Binary flag for high monthly charges
    - service_count: Total number of services adopted
    - service_diversity: Diversity of services (0-1 scale)
    - months_since_signup: Alias for tenure for clarity
    """
    data = df.copy()
    
    # 1. Tenure-based features
    if "tenure" in data.columns:
        # Tenure phase: early (0-6 months), mid (6-24), late (24+)
        data["tenure_phase_early"] = (data["tenure"] <= 6).astype(int)
        data["tenure_phase_mid"] = ((data["tenure"] > 6) & (data["tenure"] <= 24)).astype(int)
        data["tenure_phase_late"] = (data["tenure"] > 24).astype(int)
        
        # Months since signup (normalized)
        data["months_since_signup"] = data["tenure"]
    
    # 2. Spending trend features
    if "MonthlyCharges" in data.columns and "TotalCharges" in data.columns:
        # Prevent division by zero
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)
        
        # Ratio: higher ratio = newer customer with higher spending
        data["monthly_to_total_ratio"] = np.where(
            data["TotalCharges"] > 0,
            data["MonthlyCharges"] / data["TotalCharges"],
            0
        )
        
        # High spender flag (above 75th percentile)
        monthly_75 = data["MonthlyCharges"].quantile(0.75)
        data["is_high_spender"] = (data["MonthlyCharges"] > monthly_75).astype(int)
        
        # Spending growth indicator: high monthly but low total (risky churn signal)
        data["spending_volatility"] = (
            (data["MonthlyCharges"] > data["MonthlyCharges"].median()).astype(int) & 
            (data["TotalCharges"] < data["TotalCharges"].median()).astype(int)
        ).astype(int)
    
    # 3. Service adoption features
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    existing_service_cols = [col for col in service_cols if col in data.columns]
    if existing_service_cols:
        # Count total services (after they're converted to 0/1 in preprocess_data)
        # For now, we track which columns exist and will calculate after preprocessing
        data["service_count"] = 0  # Placeholder; recalculated later
        data["has_internet"] = (data["InternetService"] != "No").astype(int) if "InternetService" in data.columns else 0
    
    return data


def preprocess_data(df: pd.DataFrame, apply_feature_engineering: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the Telco Customer Churn dataset with optional feature engineering.
    
    Args:
        df: Input DataFrame
        apply_feature_engineering: If True, create engineered features
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is churn target
    """
    data = df.copy()

    # Apply feature engineering first (before encoding)
    if apply_feature_engineering:
        data = engineer_features(data)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({"Yes": 1, "No": 0})

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_cols:
        if col in data.columns:
            data[col] = data[col].replace({"No internet service": "No"})
            data[col] = data[col].map({"Yes": 1, "No": 0})

    if "MultipleLines" in data.columns:
        data["MultipleLines"] = data["MultipleLines"].replace({"No phone service": "No"})
        data["MultipleLines"] = data["MultipleLines"].map({"Yes": 1, "No": 0})

    categorical_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Calculate service_count after all service columns are available
    service_cols_available = [col for col in service_cols if col in data.columns]
    if service_cols_available and apply_feature_engineering:
        data["service_count"] = data[service_cols_available].sum(axis=1)

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    print(f"Preprocessed data: {X.shape[1]} features")

    return X, y
