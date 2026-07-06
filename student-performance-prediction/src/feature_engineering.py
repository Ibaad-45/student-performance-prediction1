"""
feature_engineering.py
------------------------
Adds derived, domain-informed features on top of the cleaned dataset.
These interaction/ratio features often carry more predictive signal than
raw columns alone and are a standard step to demonstrate in a portfolio
ML project.

New features created:
    - study_efficiency          : study hours relative to screen time distraction
    - attendance_study_index    : combined engagement score from attendance & study hours
    - academic_momentum         : blends previous score with current study habits
    - lifestyle_balance_score   : sleep vs. screen time balance

This module operates on the RAW (pre-encoding) dataframe so ratios are
computed from real units, then is called before encoding/scaling in the
pipeline.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of `df` with additional engineered columns appended."""
    df = df.copy()

    # Studying with less screen-time distraction implies higher efficiency.
    # +1 avoids division by zero for students who report 0 screen time.
    df["study_efficiency"] = (df["study_hours_per_week"] / (df["screen_time_hours"] + 1)).round(3)

    # Combined engagement: normalized attendance * normalized study hours.
    # High in both -> strong engagement signal; low in either -> pulled down.
    df["attendance_study_index"] = (
        (df["attendance_percentage"] / 100) * (df["study_hours_per_week"] / 40)
    ).round(3)

    # Blends prior achievement (70%) with current study effort (30%) to
    # capture "momentum" -- a strong past + strong present habits.
    df["academic_momentum"] = (
        0.7 * df["previous_exam_score"] + 0.3 * (df["study_hours_per_week"] / 40 * 100)
    ).round(2)

    # Healthy sleep with controlled screen time -> higher lifestyle balance.
    df["lifestyle_balance_score"] = (
        (df["sleep_hours"] / 10) - (df["screen_time_hours"] / 12)
    ).round(3)

    return df


ENGINEERED_FEATURE_NAMES = [
    "study_efficiency",
    "attendance_study_index",
    "academic_momentum",
    "lifestyle_balance_score",
]


if __name__ == "__main__":
    # Quick smoke test when run directly
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_raw_data, handle_missing_values

    df = load_raw_data()
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    print(df[ENGINEERED_FEATURE_NAMES].describe())
