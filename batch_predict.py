"""
batch_predict.py — Batch prediction for a CSV of students.

Usage:
    python batch_predict.py --input students.csv --output predictions.csv

The input CSV must contain the same feature columns as the training data.
Missing optional features are filled with sensible defaults.
"""

import os, sys, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from model_trainer  import load_models, predict_student


DEFAULTS = {
    "study_hours_per_day":       4.0,
    "attendance_pct":            75.0,
    "prev_semester_marks":       60.0,
    "assignment_completion_pct": 70.0,
    "num_subjects":              5,
    "sleep_hours":               7.0,
    "extracurricular":           0,
    "internet_access":           1,
    "family_income_level":       "Medium",
    "parental_education":        "Secondary",
    "mental_health_score":       6.0,
    "peer_group_quality":        6.0,
    "teacher_quality":           7.0,
    "library_visits_per_week":   2,
    "online_courses":            1,
    "part_time_job":             0,
    "travel_time_hours":         0.5,
}


def batch_predict(input_path: str, output_path: str):
    print(f"Loading models…")
    models = load_models()
    any_loaded = any(v is not None for v in models.values())
    if not any_loaded:
        print("ERROR: No trained models found. Run `python train_models.py` first.")
        sys.exit(1)

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} rows × {len(df.columns)} columns")

    # Fill missing columns with defaults
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            print(f"  Warning: '{col}' not found — using default {default}")

    results = []
    for idx, row in df.iterrows():
        student = {col: row[col] for col in FEATURE_COLUMNS if col in row}
        pred    = predict_student(student, models)
        results.append({
            **{col: row.get(col) for col in FEATURE_COLUMNS},
            "predicted_grade":     pred["grade"],
            "predicted_score":     round(pred["score"], 2),
            "at_risk":             pred["at_risk"],
            "risk_probability":    round(pred["risk_proba"], 4),
            "estimated_percentile": pred["percentile_est"],
            "top_suggestion":      pred["suggestions"][0] if pred["suggestions"] else "",
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Grade distribution:\n{out_df['predicted_grade'].value_counts().sort_index()}")
    print(f"At-risk students: {out_df['at_risk'].sum()} ({out_df['at_risk'].mean()*100:.1f}%)")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch student performance prediction")
    parser.add_argument("--input",  required=True,  help="Input CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()
    batch_predict(args.input, args.output)
