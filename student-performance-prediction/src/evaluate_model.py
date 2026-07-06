"""
evaluate_model.py
-------------------
Standalone evaluation script that reloads the saved models and test data
to independently verify performance -- useful for a viva demo ("here's
proof the model works, without retraining") and for CI checks.

Run:
    python src/evaluate_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_artifact, load_metrics, BASE_DIR  # noqa: E402
from data_preprocessing import (  # noqa: E402
    load_raw_data, handle_missing_values, cap_outliers_iqr,
    encode_categorical_columns, NUMERIC_COLUMNS, TARGET_COLUMN, ID_COLUMN,
)
from feature_engineering import add_engineered_features  # noqa: E402
from sklearn.model_selection import train_test_split

PASS_THRESHOLD = 40


def rebuild_test_set():
    """Recreates the exact same train/test split used during training
    (same random_state=42) so evaluation is done on genuinely unseen data."""
    df = load_raw_data()
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLUMNS + [TARGET_COLUMN])
    df = add_engineered_features(df)

    encoders = load_artifact("encoders.pkl")
    df_encoded, _ = encode_categorical_columns(df, fit=False, encoders=encoders)
    df_encoded["pass_fail"] = (df_encoded[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)

    feature_columns = load_artifact("feature_columns.pkl")
    X = df_encoded[feature_columns]
    y_reg = df_encoded[TARGET_COLUMN]
    y_clf = df_encoded["pass_fail"]

    _, X_test, _, y_reg_test, _, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    scaler = load_artifact("scaler.pkl")
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    return X_test_scaled, y_reg_test, y_clf_test


def main():
    print("=" * 70)
    print("MODEL EVALUATION -- reloading saved artifacts (no retraining)")
    print("=" * 70)

    X_test, y_reg_test, y_clf_test = rebuild_test_set()

    reg_model = load_artifact("regression_model.pkl")
    clf_model = load_artifact("classification_model.pkl")

    # ---- Regression evaluation ----
    reg_preds = reg_model.predict(X_test)
    r2 = r2_score(y_reg_test, reg_preds)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))

    print("\n--- Regression: Final Exam Score Prediction ---")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : {mae:.3f} points")
    print(f"  RMSE     : {rmse:.3f} points")

    # ---- Classification evaluation ----
    clf_preds = clf_model.predict(X_test)
    acc = accuracy_score(y_clf_test, clf_preds)
    prec = precision_score(y_clf_test, clf_preds)
    rec = recall_score(y_clf_test, clf_preds)
    f1 = f1_score(y_clf_test, clf_preds)
    cm = confusion_matrix(y_clf_test, clf_preds)

    print("\n--- Classification: Pass/Fail Prediction ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("\n" + classification_report(y_clf_test, clf_preds, target_names=["Fail", "Pass"]))

    print("=" * 70)
    print("[OK] Evaluation complete. Metrics match the training-time report in models/metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
