"""
generate_dataset.py
--------------------
Generates a realistic, statistically-sound synthetic dataset that mimics
real-world student performance datasets (e.g. UCI Student Performance,
Kaggle "Students Performance in Exams").

Why synthetic data?
Public datasets change hosts/licenses often and this keeps the project
100% reproducible offline with zero external downloads. The generation
process encodes genuine, literature-backed relationships (study time,
attendance, and prior achievement are the strongest predictors of final
grades) so every downstream step -- EDA, feature importance, model
metrics -- behaves like it would on real data.

Run:
    python src/generate_dataset.py
Produces:
    data/raw/student_performance_raw.csv
"""

import os
import numpy as np
import pandas as pd

# Fixed seed -> identical dataset every time this script runs (reproducibility)
SEED = 42
N_STUDENTS = 2500

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "student_performance_raw.csv")


def generate_dataset(n_students: int = N_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    """Builds the synthetic student performance dataframe.

    Each feature is sampled from a distribution chosen to resemble the
    real-world variable (e.g. attendance is left-skewed towards high
    values, study hours follow a mild right skew). The target variable
    `final_exam_score` is generated as a weighted combination of the
    features plus Gaussian noise, then clipped to a valid 0-100 range.
    """
    rng = np.random.default_rng(seed)

    student_id = np.arange(1, n_students + 1)

    gender = rng.choice(["Male", "Female"], size=n_students, p=[0.51, 0.49])
    age = rng.integers(15, 23, size=n_students)  # 15 to 22 inclusive

    # Study hours per week: right-skewed (most students study 5-20 hrs)
    study_hours_per_week = np.clip(rng.gamma(shape=3.0, scale=4.0, size=n_students), 0, 40).round(1)

    # Attendance: left-skewed towards high values (most students attend often)
    attendance_percentage = np.clip(rng.beta(a=6, b=2, size=n_students) * 100, 35, 100).round(1)

    # Previous exam score: roughly normal around 65
    previous_exam_score = np.clip(rng.normal(loc=65, scale=15, size=n_students), 0, 100).round(1)

    parental_education = rng.choice(
        ["High School", "Bachelors", "Masters", "PhD"],
        size=n_students, p=[0.35, 0.35, 0.22, 0.08]
    )

    family_income_level = rng.choice(["Low", "Medium", "High"], size=n_students, p=[0.30, 0.50, 0.20])

    internet_access = rng.choice(["Yes", "No"], size=n_students, p=[0.82, 0.18])

    extracurricular_activities = rng.choice(["Yes", "No"], size=n_students, p=[0.55, 0.45])

    sleep_hours = np.clip(rng.normal(loc=7, scale=1.3, size=n_students), 4, 10).round(1)

    tutoring = rng.choice(["Yes", "No"], size=n_students, p=[0.30, 0.70])

    part_time_job = rng.choice(["Yes", "No"], size=n_students, p=[0.25, 0.75])

    screen_time_hours = np.clip(rng.gamma(shape=2.2, scale=1.8, size=n_students), 0, 12).round(1)

    parental_involvement = rng.choice(["Low", "Medium", "High"], size=n_students, p=[0.25, 0.45, 0.30])

    # ---- Encode ordinal effects used purely to build the synthetic target ----
    parent_edu_bonus = pd.Series(parental_education).map(
        {"High School": 0, "Bachelors": 2, "Masters": 4, "PhD": 6}
    ).values
    income_bonus = pd.Series(family_income_level).map({"Low": -2, "Medium": 0, "High": 3}).values
    involvement_bonus = pd.Series(parental_involvement).map({"Low": -3, "Medium": 0, "High": 4}).values
    tutoring_bonus = np.where(tutoring == "Yes", 4.0, 0.0)
    extracurricular_bonus = np.where(extracurricular_activities == "Yes", 1.5, 0.0)
    job_penalty = np.where(part_time_job == "Yes", -4.0, 0.0)
    internet_bonus = np.where(internet_access == "Yes", 2.0, -1.0)

    # ---- Target generation: weighted combination + noise ----
    noise = rng.normal(loc=0, scale=6.5, size=n_students)

    final_exam_score = (
        0.34 * previous_exam_score
        + 0.26 * (study_hours_per_week / 40 * 100)
        + 0.16 * attendance_percentage
        + 0.05 * (sleep_hours / 10 * 100)
        + parent_edu_bonus
        + income_bonus
        + involvement_bonus
        + tutoring_bonus
        + extracurricular_bonus
        + job_penalty
        + internet_bonus
        - 1.1 * screen_time_hours
        + noise
    )
    final_exam_score = np.clip(final_exam_score, 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": student_id,
        "gender": gender,
        "age": age,
        "study_hours_per_week": study_hours_per_week,
        "attendance_percentage": attendance_percentage,
        "previous_exam_score": previous_exam_score,
        "parental_education": parental_education,
        "family_income_level": family_income_level,
        "internet_access": internet_access,
        "extracurricular_activities": extracurricular_activities,
        "sleep_hours": sleep_hours,
        "tutoring": tutoring,
        "part_time_job": part_time_job,
        "screen_time_hours": screen_time_hours,
        "parental_involvement": parental_involvement,
        "final_exam_score": final_exam_score,
    })

    # Inject a small, realistic proportion of missing values (~2%) into a
    # few columns so the preprocessing pipeline has genuine work to do,
    # exactly like a real-world messy dataset would.
    for col in ["attendance_percentage", "sleep_hours", "parental_education"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * n_students), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def main():
    df = generate_dataset()
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[OK] Generated {len(df)} student records -> {os.path.abspath(RAW_DATA_PATH)}")
    print(f"[OK] Missing values injected: \n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    main()
