"""
predict.py
----------
Loads the saved model artifacts once and exposes `predict_student_performance()`
-- the single function used by both the Flask API (app.py) and this script's
CLI demo. Keeping the prediction logic in one place guarantees the web app
and any command-line usage always produce identical results.

CLI usage (example):
    python src/predict.py
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_artifact, score_to_grade, grade_to_remark, PASS_THRESHOLD  # noqa: E402
from data_preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS  # noqa: E402
from feature_engineering import add_engineered_features  # noqa: E402

# Valid choices for each categorical field -- reused for input validation
# in app.py so the API and the model never disagree about accepted values.
VALID_CATEGORIES = {
    "gender": ["Male", "Female"],
    "parental_education": ["High School", "Bachelors", "Masters", "PhD"],
    "family_income_level": ["Low", "Medium", "High"],
    "internet_access": ["Yes", "No"],
    "extracurricular_activities": ["Yes", "No"],
    "tutoring": ["Yes", "No"],
    "part_time_job": ["Yes", "No"],
    "parental_involvement": ["Low", "Medium", "High"],
}

# Valid numeric ranges -- used for input validation
NUMERIC_RANGES = {
    "age": (15, 22),
    "study_hours_per_week": (0, 60),
    "attendance_percentage": (0, 100),
    "previous_exam_score": (0, 100),
    "sleep_hours": (0, 14),
    "screen_time_hours": (0, 16),
}

REQUIRED_FIELDS = list(NUMERIC_RANGES.keys()) + list(VALID_CATEGORIES.keys())


class ModelBundle:
    """Lazily loads and caches all model artifacts so repeated Flask
    requests don't hit the filesystem every time. A single instance is
    created and reused for the app's lifetime."""

    _instance = None

    def __init__(self):
        self.regression_model = load_artifact("regression_model.pkl")
        self.classification_model = load_artifact("classification_model.pkl")
        self.scaler = load_artifact("scaler.pkl")
        self.encoders = load_artifact("encoders.pkl")
        self.feature_columns = load_artifact("feature_columns.pkl")

    @classmethod
    def get_instance(cls) -> "ModelBundle":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def validate_input(data: dict):
    """Validates a raw input dict from the API/form before prediction.
    Returns a list of human-readable error strings (empty list = valid).
    Centralizing validation here means the same rules apply whether the
    request comes from the web form, a REST client, or the CLI.
    """
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data or data[field] in (None, "", []):
            errors.append(f"'{field}' is required.")

    for field, (low, high) in NUMERIC_RANGES.items():
        if field in data and data[field] not in (None, ""):
            try:
                value = float(data[field])
                if not (low <= value <= high):
                    errors.append(f"'{field}' must be between {low} and {high} (got {value}).")
            except (ValueError, TypeError):
                errors.append(f"'{field}' must be a number (got '{data[field]}').")

    for field, valid_values in VALID_CATEGORIES.items():
        if field in data and data[field] not in (None, "") and data[field] not in valid_values:
            errors.append(f"'{field}' must be one of {valid_values} (got '{data[field]}').")

    return errors


def build_feature_row(data: dict, bundle: ModelBundle) -> pd.DataFrame:
    """Transforms a validated raw input dict into the exact scaled,
    encoded feature row the trained models expect -- replaying the same
    steps used during training (feature engineering -> encoding -> scaling)."""

    row = {
        "gender": data["gender"],
        "age": float(data["age"]),
        "study_hours_per_week": float(data["study_hours_per_week"]),
        "attendance_percentage": float(data["attendance_percentage"]),
        "previous_exam_score": float(data["previous_exam_score"]),
        "parental_education": data["parental_education"],
        "family_income_level": data["family_income_level"],
        "internet_access": data["internet_access"],
        "extracurricular_activities": data["extracurricular_activities"],
        "sleep_hours": float(data["sleep_hours"]),
        "tutoring": data["tutoring"],
        "part_time_job": data["part_time_job"],
        "screen_time_hours": float(data["screen_time_hours"]),
        "parental_involvement": data["parental_involvement"],
    }
    df = pd.DataFrame([row])
    df = add_engineered_features(df)

    # Encode categoricals using the saved (already-fitted) encoders
    for col in CATEGORICAL_COLUMNS:
        le = bundle.encoders[col]
        known_classes = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else le.classes_[0])
        df[col] = le.transform(df[col])

    df = df[bundle.feature_columns]  # enforce exact training-time column order
    scaled = bundle.scaler.transform(df)
    return pd.DataFrame(scaled, columns=bundle.feature_columns)


def predict_student_performance(data: dict) -> dict:
    """Main prediction entry point.

    Args:
        data: dict of raw student features (see REQUIRED_FIELDS).

    Returns:
        dict with predicted_score, predicted_grade, pass_fail label,
        pass_probability, and a human-readable remark.

    Raises:
        ValueError: if input validation fails (caller should catch this
        and return an HTTP 400 with the error list).
    """
    errors = validate_input(data)
    if errors:
        raise ValueError("; ".join(errors))

    bundle = ModelBundle.get_instance()
    X = build_feature_row(data, bundle)

    predicted_score = float(bundle.regression_model.predict(X)[0])
    predicted_score = round(max(0.0, min(100.0, predicted_score)), 2)

    pass_fail_pred = int(bundle.classification_model.predict(X)[0])
    pass_probability = float(bundle.classification_model.predict_proba(X)[0][1])

    grade = score_to_grade(predicted_score)
    remark = grade_to_remark(grade)

    return {
        "predicted_score": predicted_score,
        "predicted_grade": grade,
        "pass_fail": "Pass" if pass_fail_pred == 1 else "Fail",
        "pass_probability": round(pass_probability * 100, 2),
        "remark": remark,
        "pass_threshold": PASS_THRESHOLD,
    }


if __name__ == "__main__":
    # CLI demo with a realistic example student
    example_student = {
        "gender": "Female",
        "age": 19,
        "study_hours_per_week": 18,
        "attendance_percentage": 91,
        "previous_exam_score": 78,
        "parental_education": "Masters",
        "family_income_level": "Medium",
        "internet_access": "Yes",
        "extracurricular_activities": "Yes",
        "sleep_hours": 7.5,
        "tutoring": "Yes",
        "part_time_job": "No",
        "screen_time_hours": 2.5,
        "parental_involvement": "High",
    }
    result = predict_student_performance(example_student)
    print("Example prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
