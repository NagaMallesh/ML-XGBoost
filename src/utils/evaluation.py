"""Evaluation and plotting utilities."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve


def find_best_threshold(
    y_true,
    y_pred_proba,
    min_precision: float | None = None,
) -> tuple[float, dict]:
    """Find the best classification threshold by maximizing F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}

    for idx, threshold in enumerate(thresholds):
        precision = float(precisions[idx + 1])
        recall = float(recalls[idx + 1])

        if min_precision is not None and precision < min_precision:
            continue

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": float(f1),
            }

    if best_metrics["f1"] < 0:
        best_threshold = 0.5
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        best_metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }

    return best_threshold, best_metrics


def evaluate_model(model, X_test, y_test, y_pred_proba=None, threshold: float = 0.5) -> dict:
    """Evaluate the model and print metrics."""
    if y_pred_proba is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Threshold: {threshold:.4f}")

    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": threshold,
    }


def plot_feature_importance(model, feature_names, output_dir: Path, top_n: int = 20) -> None:
    """Plot feature importance from XGBoost model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()

    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Feature importance plot saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    """Plot confusion matrix heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix plot saved to {output_path}")
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, output_dir: Path) -> None:
    """Plot ROC curve with AUC score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ ROC curve saved to {output_path}")
    plt.close()


def plot_precision_recall_curve(y_test, y_pred_proba, output_dir: Path, selected_threshold: float = None) -> None:
    """Plot Precision-Recall curve, optionally marking selected threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', lw=2, label='Precision-Recall curve')
    
    # Mark the selected threshold if provided
    if selected_threshold is not None:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - selected_threshold))
        plt.scatter(recalls[idx + 1], precisions[idx + 1], 
                   color='red', s=100, zorder=5, 
                   label=f'Selected (θ={selected_threshold:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    output_path = output_dir / "precision_recall_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Precision-Recall curve saved to {output_path}")
    plt.close()


def plot_class_distribution(y_train_before, y_train_after, output_dir: Path) -> None:
    """Plot class distribution before and after SMOTE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before SMOTE
    before_counts = pd.Series(y_train_before).value_counts().sort_index()
    axes[0].bar(['No Churn', 'Churn'], before_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution Before SMOTE')
    axes[0].set_ylim([0, max(before_counts.values) * 1.1])
    for i, v in enumerate(before_counts.values):
        axes[0].text(i, v + max(before_counts.values) * 0.02, str(v), ha='center', fontweight='bold')
    
    # After SMOTE
    after_counts = pd.Series(y_train_after).value_counts().sort_index()
    axes[1].bar(['No Churn', 'Churn'], after_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution After SMOTE')
    axes[1].set_ylim([0, max(after_counts.values) * 1.1])
    for i, v in enumerate(after_counts.values):
        axes[1].text(i, v + max(after_counts.values) * 0.02, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "class_distribution_smote.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Class distribution plot saved to {output_path}")
    plt.close()


def plot_threshold_vs_metrics(y_test, y_pred_proba, output_dir: Path, selected_threshold: float = None) -> None:
    """Plot how precision, recall, and F1 vary with threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, 'r-', label='F1-Score', linewidth=2)
    
    # Mark selected threshold
    if selected_threshold is not None:
        plt.axvline(x=selected_threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Selected θ={selected_threshold:.3f}')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Metrics Trade-off')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    output_path = output_dir / "threshold_vs_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Threshold vs metrics plot saved to {output_path}")
    plt.close()


def plot_learning_curves(model, X_train, y_train, output_dir: Path, cv: int = 5) -> None:
    """Plot learning curves to diagnose overfitting/underfitting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adapt CV folds based on training set size
    n_samples = len(X_train)
    if n_samples < 10:
        print(f"⚠ Warning: Skipping learning curves - insufficient samples ({n_samples} < 10)")
        return
    
    # Check for sufficient minority class samples
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    min_class_count = min(counts) if len(counts) > 1 else counts[0]
    
    # Use at most n_samples folds, minimum 2
    cv = min(cv, n_samples // 2, 10)
    cv = max(cv, 2)
    
    # Need at least cv samples per class for stratified CV
    if min_class_count < cv:
        print(f"⚠ Warning: Skipping learning curves - insufficient minority class samples "
              f"({min_class_count} samples, need {cv} for {cv}-fold CV)")
        return
    
    print(f"\nGenerating learning curves (this may take a moment, using {cv}-fold stratified CV)...")
    
    # Clone model without early stopping (incompatible with learning_curve CV)
    try:
        from copy import deepcopy
        model_for_cv = deepcopy(model)
        # Remove early stopping to avoid validation set requirement
        if hasattr(model_for_cv, 'early_stopping_rounds'):
            model_for_cv.set_params(early_stopping_rounds=None)
    except Exception:
        # If cloning fails, use original model and handle potential errors
        model_for_cv = model
    
    try:
        from sklearn.model_selection import StratifiedKFold
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Use stratified CV to ensure both classes in each fold
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model_for_cv, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv_splitter,
            scoring='f1',
            n_jobs=-1
        )
    except ValueError as e:
        error_msg = str(e).lower()
        if "early stopping" in error_msg:
            print(f"⚠ Warning: Skipping learning curves - model incompatible with CV (early stopping)")
            return
        elif "pos_label" in error_msg or "not a valid label" in error_msg:
            print(f"⚠ Warning: Skipping learning curves - class imbalance too extreme for CV")
            return
        else:
            raise
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='green', label='Cross-validation score', linewidth=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Learning curves saved to {output_path}")
    plt.close()


def plot_calibration_curve(y_test, y_pred_proba, output_dir: Path, n_bins: int = 10) -> None:
    """Plot calibration curve to assess probability quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='XGBoost', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (True Probability)')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    
    output_path = output_dir / "calibration_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Calibration curve saved to {output_path}")
    plt.close()


def plot_feature_correlation_heatmap(X, output_dir: Path, top_n: int = 20) -> None:
    """Plot correlation heatmap for top features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Select top N features by average absolute correlation
    avg_corr = corr_matrix.abs().mean().sort_values(ascending=False)
    top_features = avg_corr.head(top_n).index
    corr_subset = corr_matrix.loc[top_features, top_features]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title(f'Feature Correlation Heatmap (Top {top_n} Features)')
    plt.tight_layout()
    
    output_path = output_dir / "feature_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Feature correlation heatmap saved to {output_path}")
    plt.close()


def plot_churn_rate_by_feature(df, feature_col: str, output_dir: Path) -> None:
    """Plot churn rate by categorical feature values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if feature_col not in df.columns or 'Churn' not in df.columns:
        print(f"⚠ Warning: Cannot plot churn rate for {feature_col} - column not found")
        return
    
    # Calculate churn rate by feature
    churn_rate = df.groupby(feature_col)['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) if len(x) > 0 else 0
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(churn_rate)), churn_rate.values, color='coral')
    plt.xticks(range(len(churn_rate)), churn_rate.index, rotation=45, ha='right')
    plt.ylabel('Churn Rate')
    plt.xlabel(feature_col)
    plt.title(f'Churn Rate by {feature_col}')
    plt.ylim([0, max(churn_rate.values) * 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(churn_rate.values):
        plt.text(i, v + max(churn_rate.values) * 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f"churn_rate_by_{feature_col.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Churn rate by {feature_col} saved to {output_path}")
    plt.close()