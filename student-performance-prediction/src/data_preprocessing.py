"""
data_preprocessing.py
----------------------
Handles all data cleaning for the student performance dataset:
    1. Loading the raw CSV
    2. Missing value imputation (median for numeric, mode for categorical)
    3. Outlier capping using the IQR method
    4. Categorical encoding (Label Encoding for ordinal-ish columns)
    5. Train/test split
    6. Feature scaling (StandardScaler, fit on train only to avoid leakage)

Every fitted transformer (encoders, scaler) is saved so the exact same
transformation can be replayed at inference time in predict.py / app.py.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_artifact, BASE_DIR  # noqa: E402
from feature_engineering import add_engineered_features  # noqa: E402

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "student_performance_raw.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CATEGORICAL_COLUMNS = [
    "gender", "parental_education", "family_income_level", "internet_access",
    "extracurricular_activities", "tutoring", "part_time_job", "parental_involvement",
]
NUMERIC_COLUMNS = [
    "age", "study_hours_per_week", "attendance_percentage", "previous_exam_score",
    "sleep_hours", "screen_time_hours",
]
TARGET_COLUMN = "final_exam_score"
ID_COLUMN = "student_id"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw CSV, raising a clear error if the dataset hasn't
    been generated yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python src/generate_dataset.py` first."
        )
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values:
    - Numeric columns -> median (robust to outliers/skew)
    - Categorical columns -> mode (most frequent category)
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Caps (winsorizes) outliers using the 1.5*IQR rule so a handful of
    extreme values don't distort model training, while preserving all rows
    (capping is preferred over dropping for a small dataset)."""
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def encode_categorical_columns(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """Label-encodes each categorical column. When `fit=True`, new
    LabelEncoders are fitted and returned; when `fit=False`, the passed-in
    `encoders` dict (loaded from disk) is reused so unseen inference-time
    data is transformed identically to training-time data."""
    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Map unseen categories at inference time to the most common
            # training class instead of crashing, for robustness.
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders


def run_preprocessing_pipeline(save_outputs: bool = True):
    """End-to-end preprocessing: load -> clean -> encode -> split -> scale.
    Returns X_train, X_test, y_train, y_test, y_train_pass_fail, y_test_pass_fail
    plus the fitted encoders and scaler.
    """
    from utils import PASS_THRESHOLD

    df = load_raw_data()
    print(f"[1/6] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = handle_missing_values(df)
    print(f"[2/6] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

    df = cap_outliers_iqr(df, NUMERIC_COLUMNS + [TARGET_COLUMN])
    print("[3/6] Outliers capped using IQR method")

    df = add_engineered_features(df)
    print("[3.5/6] Added engineered features (study_efficiency, academic_momentum, etc.)")

    df, encoders = encode_categorical_columns(df, fit=True)
    print(f"[4/6] Encoded {len(CATEGORICAL_COLUMNS)} categorical columns")

    # Classification target: Pass (1) if final_exam_score >= threshold, else Fail (0)
    df["pass_fail"] = (df[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)

    feature_columns = [c for c in df.columns if c not in [ID_COLUMN, TARGET_COLUMN, "pass_fail"]]

    X = df[feature_columns]
    y_reg = df[TARGET_COLUMN]
    y_clf = df["pass_fail"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    print(f"[5/6] Split into train ({X_train.shape[0]}) / test ({X_test.shape[0]})")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_columns, index=X_test.index
    )
    print("[6/6] Features scaled with StandardScaler (fit on train only)")

    if save_outputs:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        train_out = X_train.copy()
        train_out[TARGET_COLUMN] = y_reg_train
        train_out["pass_fail"] = y_clf_train
        train_out.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)

        test_out = X_test.copy()
        test_out[TARGET_COLUMN] = y_reg_test
        test_out["pass_fail"] = y_clf_test
        test_out.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

        save_artifact(encoders, "encoders.pkl")
        save_artifact(scaler, "scaler.pkl")
        save_artifact(feature_columns, "feature_columns.pkl")
        print(f"[OK] Processed data saved to {PROCESSED_DIR}")
        print("[OK] Encoders, scaler, and feature column order saved to models/")

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
        "encoders": encoders, "scaler": scaler,
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
