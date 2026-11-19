"""
Student Academic Performance Predictor
Data generation and feature engineering module.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for student academic performance prediction.

    Features:
        - study_hours_per_day       : Daily study hours (1–10)
        - attendance_pct            : Attendance percentage (40–100)
        - prev_semester_marks       : Previous semester GPA out of 100 (30–100)
        - assignment_completion_pct : Assignment completion rate (30–100)
        - num_subjects              : Number of subjects enrolled (3–8)
        - sleep_hours               : Average sleep hours per night (4–10)
        - extracurricular           : Participation in extracurricular activities (0/1)
        - internet_access           : Has reliable internet access at home (0/1)
        - family_income_level       : Family income bracket (Low/Medium/High)
        - parental_education        : Highest parental education (None/Primary/Secondary/Graduate/Postgrad)
        - mental_health_score       : Self-reported mental-health score 1–10
        - peer_group_quality        : Quality of peer group 1–10
        - teacher_quality           : Perceived teacher quality 1–10
        - library_visits_per_week   : Library visits per week (0–7)
        - online_courses            : Number of online/MOOC courses enrolled (0–5)
        - part_time_job             : Has a part-time job (0/1)
        - travel_time_hours         : Daily commute time in hours (0–3)
        - performance_grade         : Target — Grade category A/B/C/D/F
        - performance_score         : Target — Continuous score 0–100
    """
    rng = np.random.default_rng(random_state)

    study_hours      = rng.uniform(1, 10, n_samples)
    attendance_pct   = rng.uniform(40, 100, n_samples)
    prev_marks       = rng.uniform(30, 100, n_samples)
    assignment_pct   = rng.uniform(30, 100, n_samples)
    num_subjects     = rng.integers(3, 9, n_samples)
    sleep_hours      = rng.uniform(4, 10, n_samples)
    extracurricular  = rng.integers(0, 2, n_samples)
    internet_access  = rng.integers(0, 2, n_samples)
    family_income    = rng.choice(["Low", "Medium", "High"], n_samples, p=[0.3, 0.45, 0.25])
    parental_edu     = rng.choice(["None", "Primary", "Secondary", "Graduate", "Postgrad"],
                                   n_samples, p=[0.05, 0.10, 0.35, 0.35, 0.15])
    mental_health    = rng.uniform(1, 10, n_samples)
    peer_quality     = rng.uniform(1, 10, n_samples)
    teacher_quality  = rng.uniform(1, 10, n_samples)
    library_visits   = rng.integers(0, 8, n_samples)
    online_courses   = rng.integers(0, 6, n_samples)
    part_time_job    = rng.integers(0, 2, n_samples)
    travel_time      = rng.uniform(0, 3, n_samples)

    # Income numeric mapping
    income_map = {"Low": 0, "Medium": 1, "High": 2}
    income_num = np.array([income_map[x] for x in family_income])

    # Education numeric mapping
    edu_map = {"None": 0, "Primary": 1, "Secondary": 2, "Graduate": 3, "Postgrad": 4}
    edu_num = np.array([edu_map[x] for x in parental_edu])

    # Composite performance score (domain-driven formula)
    score = (
        study_hours      * 4.0
        + attendance_pct  * 0.25
        + prev_marks      * 0.25
        + assignment_pct  * 0.15
        + sleep_hours     * 1.5
        + extracurricular * 2.5
        + internet_access * 2.0
        + income_num      * 1.5
        + edu_num         * 1.5
        + mental_health   * 1.2
        + peer_quality    * 1.0
        + teacher_quality * 1.2
        + library_visits  * 1.0
        + online_courses  * 1.5
        - part_time_job   * 2.0
        - travel_time     * 1.5
        - num_subjects    * 0.5          # more subjects → more load
    )

    # Normalize to 0–100
    score_min, score_max = score.min(), score.max()
    score_norm = (score - score_min) / (score_max - score_min) * 100

    # Add mild noise
    noise = rng.normal(0, 3, n_samples)
    performance_score = np.clip(score_norm + noise, 0, 100)

    # Grade categories
    def score_to_grade(s):
        if s >= 85: return "A"
        elif s >= 70: return "B"
        elif s >= 55: return "C"
        elif s >= 40: return "D"
        else: return "F"

    performance_grade = np.array([score_to_grade(s) for s in performance_score])

    # Risk flag
    at_risk = (performance_score < 50).astype(int)

    df = pd.DataFrame({
        "study_hours_per_day":       np.round(study_hours, 1),
        "attendance_pct":            np.round(attendance_pct, 1),
        "prev_semester_marks":       np.round(prev_marks, 1),
        "assignment_completion_pct": np.round(assignment_pct, 1),
        "num_subjects":              num_subjects,
        "sleep_hours":               np.round(sleep_hours, 1),
        "extracurricular":           extracurricular,
        "internet_access":           internet_access,
        "family_income_level":       family_income,
        "parental_education":        parental_edu,
        "mental_health_score":       np.round(mental_health, 1),
        "peer_group_quality":        np.round(peer_quality, 1),
        "teacher_quality":           np.round(teacher_quality, 1),
        "library_visits_per_week":   library_visits,
        "online_courses":            online_courses,
        "part_time_job":             part_time_job,
        "travel_time_hours":         np.round(travel_time, 1),
        "performance_score":         np.round(performance_score, 2),
        "performance_grade":         performance_grade,
        "at_risk":                   at_risk,
    })
    return df


FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_pct",
    "prev_semester_marks",
    "assignment_completion_pct",
    "num_subjects",
    "sleep_hours",
    "extracurricular",
    "internet_access",
    "family_income_level",
    "parental_education",
    "mental_health_score",
    "peer_group_quality",
    "teacher_quality",
    "library_visits_per_week",
    "online_courses",
    "part_time_job",
    "travel_time_hours",
]

CATEGORICAL_COLUMNS = ["family_income_level", "parental_education"]
NUMERIC_COLUMNS     = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

GRADE_ORDER   = ["A", "B", "C", "D", "F"]
GRADE_COLORS  = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}

FEATURE_DESCRIPTIONS = {
    "study_hours_per_day":       "Daily Study Hours",
    "attendance_pct":            "Attendance %",
    "prev_semester_marks":       "Previous Semester Marks",
    "assignment_completion_pct": "Assignment Completion %",
    "num_subjects":              "Number of Subjects",
    "sleep_hours":               "Sleep Hours/Night",
    "extracurricular":           "Extracurricular Activities",
    "internet_access":           "Internet Access at Home",
    "family_income_level":       "Family Income Level",
    "parental_education":        "Parental Education Level",
    "mental_health_score":       "Mental Health Score (1–10)",
    "peer_group_quality":        "Peer Group Quality (1–10)",
    "teacher_quality":           "Teacher Quality (1–10)",
    "library_visits_per_week":   "Library Visits/Week",
    "online_courses":            "Online Courses Enrolled",
    "part_time_job":             "Has Part-Time Job",
    "travel_time_hours":         "Daily Travel Time (hrs)",
}
