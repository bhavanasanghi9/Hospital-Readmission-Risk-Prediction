from __future__ import annotations

from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from sklearn.model_selection import cross_val_score



def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba,
) -> Dict[str, float]:
    """
    Compute core binary classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_at_threshold(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Convert predicted probabilities to class predictions at a given threshold
    and compute classification metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    metrics["threshold"] = threshold
    return metrics


def build_results_row(
    model_name: str,
    threshold: float,
    y_true,
    y_pred,
    y_proba,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame summarizing model performance.
    Useful for comparison tables across models.
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    row = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "ROC_AUC": metrics["roc_auc"],
    }

    return pd.DataFrame([row])


def plot_confusion_matrix_from_preds(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
):
    """
    Plot confusion matrix from predicted labels.
    """
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.show()


def plot_roc_curve_from_proba(
    y_true,
    y_proba,
    title: str = "ROC Curve",
):
    """
    Plot ROC curve from predicted probabilities.
    """
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.show()

from sklearn.model_selection import cross_val_score


def compute_train_test_metrics(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Fit-independent evaluation on both train and test sets at a fixed threshold.
    Assumes model is already fitted.
    """
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    train_metrics = compute_classification_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_proba)

    rows = [
        {
            "Split": "Train",
            "Threshold": threshold,
            "Accuracy": train_metrics["accuracy"],
            "Precision": train_metrics["precision"],
            "Recall": train_metrics["recall"],
            "F1": train_metrics["f1"],
            "ROC_AUC": train_metrics["roc_auc"],
        },
        {
            "Split": "Test",
            "Threshold": threshold,
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1": test_metrics["f1"],
            "ROC_AUC": test_metrics["roc_auc"],
        },
    ]

    return pd.DataFrame(rows)


def compute_cv_auc(
    model,
    X,
    y,
    cv,
) -> Dict[str, float]:
    """
    Compute cross-validated ROC-AUC summary.
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )

    return {
        "cv_auc_mean": scores.mean(),
        "cv_auc_std": scores.std(),
    }

from sklearn.metrics import precision_recall_curve



def build_threshold_table(y_true, y_proba) -> pd.DataFrame:
    """
    Build a threshold table with precision, recall, and F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
    })

    threshold_df["f1"] = (
        2 * threshold_df["precision"] * threshold_df["recall"] /
        (threshold_df["precision"] + threshold_df["recall"])
    )

    threshold_df["f1"] = threshold_df["f1"].fillna(0)

    return threshold_df


def find_best_f1_threshold(y_true, y_proba) -> Dict[str, float]:
    """
    Find the threshold that maximizes F1 score.
    """
    threshold_df = build_threshold_table(y_true, y_proba)
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]

    return {
        "threshold": float(best_row["threshold"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }


def plot_probability_distribution_by_class(
    y_true,
    y_proba,
    title: str = "Probability Distribution by Class",
):
    """
    Plot predicted probability distributions for the two classes.
    """
    plt.figure(figsize=(7, 5))

    plt.hist(
        y_proba[y_true == 0],
        bins=50,
        alpha=0.6,
        label="No Readmission",
    )

    plt.hist(
        y_proba[y_true == 1],
        bins=50,
        alpha=0.6,
        label="Readmission",
    )

    plt.legend()
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

