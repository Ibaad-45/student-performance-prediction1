"""
utils.py
--------
Small, shared helper functions used across the preprocessing, training,
evaluation, and prediction scripts. Kept dependency-free (stdlib + joblib)
so it can be safely imported everywhere without circular imports.
"""

import os
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def save_artifact(obj, filename: str):
    """Serializes any Python object (model, scaler, encoder dict) to
    the models/ directory using joblib, which handles numpy arrays and
    scikit-learn objects efficiently."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(filename: str):
    """Loads a previously saved artifact from the models/ directory.
    Raises a clear FileNotFoundError if training hasn't been run yet."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact '{filename}' not found at {path}. "
            f"Run `python src/train_model.py` first to train and save the models."
        )
    return joblib.load(path)


def save_metrics(metrics: dict, filename: str = "metrics.json"):
    """Persists evaluation metrics as JSON so the Flask dashboard can
    display them without needing to re-run the training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(filename: str = "metrics.json") -> dict:
    """Loads the saved metrics JSON. Returns an empty dict (rather than
    raising) if metrics haven't been generated yet, so the dashboard
    can render a friendly empty state instead of crashing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def score_to_grade(score: float) -> str:
    """Converts a numeric final exam score (0-100) into a letter grade.
    Used consistently across training (for the classification target),
    the dashboard, and the prediction API so grade boundaries never drift."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    elif score >= 40:
        return "E"
    else:
        return "F"


def grade_to_remark(grade: str) -> str:
    """Human-readable remark shown alongside the predicted grade."""
    remarks = {
        "A+": "Outstanding performance. Keep up the excellent work!",
        "A": "Excellent performance with strong fundamentals.",
        "B": "Good performance. Minor improvements can push this higher.",
        "C": "Satisfactory performance. Consistent effort will help.",
        "D": "Below average. Consider increasing study hours and attendance.",
        "E": "At risk of falling behind. Extra support is recommended.",
        "F": "Failing trajectory. Immediate academic intervention advised.",
    }
    return remarks.get(grade, "No remark available.")


PASS_THRESHOLD = 40  # final_exam_score >= 40 is considered a "Pass"
