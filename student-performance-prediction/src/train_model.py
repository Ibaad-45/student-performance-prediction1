"""
train_model.py
---------------
End-to-end training script:
    1. Loads + cleans raw data, applies feature engineering
    2. Generates Exploratory Data Analysis (EDA) plots
    3. Encodes/scales features via the preprocessing pipeline
    4. Trains multiple regression models to predict `final_exam_score`
       and picks the best one by R^2 on the held-out test set
    5. Trains a classification model to predict Pass/Fail
    6. Generates evaluation plots (feature importance, confusion matrix,
       actual-vs-predicted scatter, residuals)
    7. Saves the winning models + a metrics.json report used by the
       Flask dashboard

Run:
    python src/train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend -- safe for servers / no display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_artifact, save_metrics, score_to_grade, BASE_DIR  # noqa: E402
from data_preprocessing import (  # noqa: E402
    load_raw_data, handle_missing_values, cap_outliers_iqr,
    encode_categorical_columns, NUMERIC_COLUMNS, TARGET_COLUMN, ID_COLUMN,
)
from feature_engineering import add_engineered_features, ENGINEERED_FEATURE_NAMES  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
# Brand palette matched to the frontend's brass/ink-green theme
BRAND_PALETTE = ["#1F3A2E", "#C89B3C", "#3D6E8C", "#B4453B", "#7A8C74"]
PASS_THRESHOLD = 40


def run_eda(df: pd.DataFrame):
    """Generates and saves exploratory data analysis charts as PNGs
    consumed by the dashboard page."""

    # 1. Correlation heatmap of numeric + engineered features vs target
    numeric_for_corr = NUMERIC_COLUMNS + ENGINEERED_FEATURE_NAMES + [TARGET_COLUMN]
    corr = df[numeric_for_corr].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, square=True,
                linewidths=0.5, linecolor="white")
    plt.title("Correlation Heatmap: Numeric Features vs Final Exam Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=140)
    plt.close()

    # 2. Distribution of the target variable
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins=30, kde=True, color=BRAND_PALETTE[0])
    plt.axvline(PASS_THRESHOLD, color=BRAND_PALETTE[3], linestyle="--", linewidth=2,
                label=f"Pass threshold ({PASS_THRESHOLD})")
    plt.title("Distribution of Final Exam Scores", fontsize=13, fontweight="bold")
    plt.xlabel("Final Exam Score")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_distribution.png"), dpi=140)
    plt.close()

    # 3. Study hours vs final score, colored by pass/fail
    plt.figure(figsize=(8, 5))
    pass_fail_labels = np.where(df[TARGET_COLUMN] >= PASS_THRESHOLD, "Pass", "Fail")
    sns.scatterplot(x=df["study_hours_per_week"], y=df[TARGET_COLUMN], hue=pass_fail_labels,
                     palette={"Pass": BRAND_PALETTE[1], "Fail": BRAND_PALETTE[3]}, alpha=0.6, s=35)
    plt.title("Study Hours per Week vs Final Exam Score", fontsize=13, fontweight="bold")
    plt.xlabel("Study Hours per Week")
    plt.ylabel("Final Exam Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "study_hours_vs_score.png"), dpi=140)
    plt.close()

    print("[EDA] Saved correlation_heatmap.png, score_distribution.png, study_hours_vs_score.png")


def train_regression_models(X_train, X_test, y_train, y_test):
    """Trains three candidate regressors, evaluates each on the test
    set, and returns the best-performing model along with all metrics
    for transparent comparison in the README/dashboard."""

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
    }

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"r2_score": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}
        fitted_models[name] = model
        print(f"[Regression] {name:28s} R2={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    best_name = max(results, key=lambda k: results[k]["r2_score"])
    print(f"[Regression] Best model: {best_name}")
    return fitted_models[best_name], best_name, results


def train_classification_model(X_train, X_test, y_train, y_test):
    """Trains a Random Forest classifier for the Pass/Fail task and
    returns the fitted model plus its evaluation metrics."""
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
    }
    print(f"[Classification] RandomForestClassifier -> {metrics}")
    print(classification_report(y_test, preds, target_names=["Fail", "Pass"]))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], linewidths=1, linecolor="white")
    plt.title("Confusion Matrix: Pass / Fail Prediction", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=140)
    plt.close()

    return model, metrics


def plot_feature_importance(model, feature_columns, model_label: str):
    """Plots and saves a horizontal bar chart of feature importances for
    tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 7))
    importances.tail(15).plot(kind="barh", color=BRAND_PALETTE[0])
    plt.title(f"Top Feature Importances ({model_label})", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=140)
    plt.close()
    print("[Plot] Saved feature_importance.png")


def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.4, color=BRAND_PALETTE[2], s=25)
    lims = [0, 100]
    plt.plot(lims, lims, color=BRAND_PALETTE[3], linestyle="--", linewidth=2, label="Perfect prediction")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Final Exam Score")
    plt.ylabel("Predicted Final Exam Score")
    plt.title("Actual vs Predicted Scores (Test Set)", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=140)
    plt.close()
    print("[Plot] Saved actual_vs_predicted.png")


def main():
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION -- MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ---- 1. Load + clean ----
    df = load_raw_data()
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLUMNS + [TARGET_COLUMN])

    # ---- 2. Feature engineering (on raw units, before encoding) ----
    df = add_engineered_features(df)

    # ---- 3. EDA plots ----
    run_eda(df)

    # ---- 4. Encode categoricals ----
    df_encoded, encoders = encode_categorical_columns(df, fit=True)
    df_encoded["pass_fail"] = (df_encoded[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)

    feature_columns = [c for c in df_encoded.columns if c not in [ID_COLUMN, TARGET_COLUMN, "pass_fail"]]
    X = df_encoded[feature_columns]
    y_reg = df_encoded[TARGET_COLUMN]
    y_clf = df_encoded["pass_fail"]

    # ---- 5. Split ----
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # ---- 6. Scale ----
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    # ---- 6.5 Persist processed train/test splits for transparency & reuse ----
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_out = X_train.copy()
    train_out[TARGET_COLUMN] = y_reg_train
    train_out["pass_fail"] = y_clf_train
    train_out.to_csv(os.path.join(processed_dir, "train.csv"), index=False)

    test_out = X_test.copy()
    test_out[TARGET_COLUMN] = y_reg_test
    test_out["pass_fail"] = y_clf_test
    test_out.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    print(f"[Data] Saved processed train/test splits -> {processed_dir}")

    # ---- 7. Train regression models ----
    best_reg_model, best_reg_name, reg_results = train_regression_models(
        X_train_scaled, X_test_scaled, y_reg_train, y_reg_test
    )
    reg_preds = best_reg_model.predict(X_test_scaled)
    plot_actual_vs_predicted(y_reg_test, reg_preds)
    plot_feature_importance(best_reg_model, feature_columns, best_reg_name)

    # ---- 8. Train classification model ----
    clf_model, clf_metrics = train_classification_model(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)

    # ---- 9. Save all artifacts ----
    save_artifact(best_reg_model, "regression_model.pkl")
    save_artifact(clf_model, "classification_model.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(encoders, "encoders.pkl")
    save_artifact(feature_columns, "feature_columns.pkl")

    metrics_report = {
        "regression": {
            "best_model": best_reg_name,
            "all_models_compared": reg_results,
            "test_r2_score": reg_results[best_reg_name]["r2_score"],
            "test_mae": reg_results[best_reg_name]["mae"],
            "test_rmse": reg_results[best_reg_name]["rmse"],
        },
        "classification": {
            "model": "RandomForestClassifier",
            **clf_metrics,
        },
        "dataset": {
            "total_students": int(len(df)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "num_features": len(feature_columns),
            "pass_rate_pct": round(float((df_encoded["pass_fail"].sum() / len(df_encoded)) * 100), 2),
        },
    }
    save_metrics(metrics_report)

    print("=" * 70)
    print("[DONE] All models, encoders, scaler, and metrics saved to /models")
    print(f"[DONE] Best regression model: {best_reg_name} (R2={reg_results[best_reg_name]['r2_score']})")
    print(f"[DONE] Classification accuracy: {clf_metrics['accuracy']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
