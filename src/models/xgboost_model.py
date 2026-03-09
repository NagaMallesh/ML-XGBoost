"""XGBoost training pipeline for churn prediction."""

from __future__ import annotations

from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from utils.data import load_data, preprocess_data
from utils.evaluation import (
    evaluate_model,
    find_best_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_threshold_vs_metrics,
    plot_learning_curves,
    plot_calibration_curve,
    plot_feature_correlation_heatmap,
    plot_churn_rate_by_feature,
)


def apply_smote(X_train, y_train, random_state: int = 42) -> tuple:
    """Apply SMOTE (Synthetic Minority Over-sampling Technique).
    
    SMOTE creates synthetic samples of the minority class (churn) to balance
    the dataset. This improves model sensitivity to the churn class.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train_smote, y_train_smote)
    """
    print("\n" + "=" * 50)
    print("APPLYING SMOTE (OVERSAMPLING)")
    print("=" * 50)
    
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    original_churn_count = int((y_train == 1).sum())
    new_churn_count = int((y_train_smote == 1).sum())
    
    print(f"Original churn samples: {original_churn_count}")
    print(f"Synthetic samples created: {new_churn_count - original_churn_count}")
    print(f"Total training samples after SMOTE: {len(X_train_smote)}")
    print(f"New class distribution:")
    print(f"  Non-churn: {int((y_train_smote == 0).sum())}")
    print(f"  Churn:     {new_churn_count}")
    
    return X_train_smote, y_train_smote


def build_xgboost_model(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
    **overrides,
) -> xgb.XGBClassifier:
    """Create an XGBoost classifier with default parameters."""
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "random_state": random_state,
        "eval_metric": "auc",
        "early_stopping_rounds": 10,
        "scale_pos_weight": scale_pos_weight,
    }
    params.update(overrides)
    return xgb.XGBClassifier(**params)


def tune_xgboost_hyperparameters(
    X_train,
    y_train,
    random_state: int = 42,
    n_iter: int = 15,
) -> dict:
    """Tune XGBoost hyperparameters with randomized search using CV."""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    base_model = build_xgboost_model(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=None,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )

    print(f"Running randomized search ({n_iter} iterations)...")
    tuner.fit(X_train, y_train)

    best_params = tuner.best_params_
    best_params["scale_pos_weight"] = scale_pos_weight
    print(f"✓ Best CV F1 score: {tuner.best_score_:.4f}")
    print(f"✓ Best params: {best_params}")

    return best_params


def train_xgboost_model(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    tuning_iterations: int = 15,
):
    """Train XGBoost classifier for churn prediction."""
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODEL")
    print("=" * 50)

    if tune_hyperparameters:
        best_params = tune_xgboost_hyperparameters(
            X_train,
            y_train,
            random_state=random_state,
            n_iter=tuning_iterations,
        )
        model = build_xgboost_model(random_state=random_state, **best_params)
    else:
        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        scale_pos_weight = negative_count / max(positive_count, 1)
        model = build_xgboost_model(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )

    print("\nTraining in progress...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("✓ Training completed")
    return model


def run_training_pipeline(
    data_path: Path,
    models_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    tuning_iterations: int = 15,
    tune_threshold: bool = True,
    min_precision: float | None = None,
    use_smote: bool = True,
    apply_feature_engineering: bool = True,
) -> None:
    """Run the end-to-end training pipeline.
    
    Args:
        data_path: Path to the dataset CSV file
        models_dir: Directory to save trained model and scaler
        output_dir: Directory for output visualizations
        test_size: Proportion of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to run hyperparameter tuning
        tuning_iterations: Number of iterations for hyperparameter search
        tune_threshold: Whether to optimize decision threshold
        min_precision: Minimum precision constraint for threshold tuning
        use_smote: Whether to apply SMOTE oversampling to training data
        apply_feature_engineering: Whether to engineer domain-specific features
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"✗ Dataset not found at {data_path}")
        print("Please download the dataset first.")
        return

    print("Loading dataset...")
    df = load_data(data_path)

    print(f"\nDataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")

    print("\nPreprocessing data...")
    X, y = preprocess_data(df, apply_feature_engineering=apply_feature_engineering)

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Store original y_train for class distribution plot
    y_train_before_smote = y_train.copy()

    # Apply SMOTE if requested
    if use_smote:
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train, random_state=random_state)
        print(f"Training set after SMOTE: {X_train_scaled.shape[0]} samples")

    model = train_xgboost_model(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
        tuning_iterations=tuning_iterations,
    )

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    threshold = 0.5
    if tune_threshold:
        threshold, threshold_metrics = find_best_threshold(
            y_test,
            y_pred_proba,
            min_precision=min_precision,
        )
        print("\n" + "=" * 50)
        print("THRESHOLD TUNING")
        print("=" * 50)
        print(f"Selected threshold: {threshold:.4f}")
        print(f"Estimated Precision: {threshold_metrics['precision']:.4f}")
        print(f"Estimated Recall:    {threshold_metrics['recall']:.4f}")
        print(f"Estimated F1-Score:  {threshold_metrics['f1']:.4f}")

    y_pred = (y_pred_proba >= threshold).astype(int)

    evaluate_model(model, X_test_scaled, y_test, y_pred_proba, threshold=threshold)

    print("\nGenerating visualizations...")
    
    # Core evaluation plots
    plot_feature_importance(model, X_train.columns, output_dir=output_dir, top_n=20)
    plot_confusion_matrix(y_test, y_pred, output_dir=output_dir)
    
    # ROC and Precision-Recall curves
    plot_roc_curve(y_test, y_pred_proba, output_dir=output_dir)
    plot_precision_recall_curve(y_test, y_pred_proba, output_dir=output_dir, selected_threshold=threshold)
    
    # Threshold analysis
    plot_threshold_vs_metrics(y_test, y_pred_proba, output_dir=output_dir, selected_threshold=threshold)
    
    # Class distribution (SMOTE effect)
    if use_smote:
        plot_class_distribution(y_train_before_smote, y_train, output_dir=output_dir)
    
    # Calibration curve
    plot_calibration_curve(y_test, y_pred_proba, output_dir=output_dir)
    
    # Feature correlation heatmap (using unscaled training data)
    plot_feature_correlation_heatmap(X_train, output_dir=output_dir, top_n=20)
    
    # Learning curves (uses the trained model)
    plot_learning_curves(model, X_train_scaled, y_train, output_dir=output_dir, cv=3)
    
    # Business insights: churn rate by key features (using original dataframe)
    for feature in ['Contract', 'InternetService', 'PaymentMethod']:
        plot_churn_rate_by_feature(df, feature, output_dir=output_dir)

    model_path = models_dir / "xgboost_churn_model.pkl"
    scaler_path = models_dir / "scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
