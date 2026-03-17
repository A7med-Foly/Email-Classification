"""
src/model_evaluation.py
Compute classification metrics and select the best model.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    return {
        "Model":     model_name or type(model).__name__,
        "Accuracy":  round(accuracy,  4),
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "CM":        cm,        # kept for display; excluded from CSV
    }


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append(m)
        logger.info(
            "%-20s | Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
            name, m["Accuracy"], m["Precision"], m["Recall"], m["F1"],
        )

    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def select_best_model(trained_models: dict, X_test, y_test) -> tuple:
    best_name, best_model, best_f1 = None, None, -1
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test)
        if m["F1"] > best_f1:
            best_f1, best_name, best_model = m["F1"], name, model
    logger.info("Best model: %s (F1=%.4f)", best_name, best_f1)
    return best_name, best_model


def save_results(results_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.drop(columns=["CM"], errors="ignore").to_csv(path, index=False)
    logger.info("Results saved to '%s'.", path)


def print_results(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON  (sorted by F1)")
    print("=" * 65)
    print(results_df.drop(columns=["CM"], errors="ignore").to_string(index=False))
    print("=" * 65 + "\n")
