"""
train_models.py — Standalone script to pre-train and save all models.

Run once before launching the Streamlit app:
    python train_models.py

This generates 2000 synthetic student records, trains all models,
evaluates them, and saves the best models to the ./models/ directory.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_synthetic_dataset
from model_trainer  import train_all_models
import pandas as pd
import json


def main():
    print("=" * 60)
    print("  Student Academic Performance Predictor — Model Training")
    print("=" * 60)

    print("\n[1/3] Generating synthetic dataset (n=2000)…")
    df = generate_synthetic_dataset(n_samples=2000, random_state=42)
    print(f"      Dataset shape : {df.shape}")
    print(f"      Grade counts  :\n{df['performance_grade'].value_counts().sort_index().to_string()}")
    print(f"      Score stats   : mean={df['performance_score'].mean():.1f}, std={df['performance_score'].std():.1f}")
    print(f"      At-risk       : {df['at_risk'].sum()} ({df['at_risk'].mean()*100:.1f}%)")

    # Save dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "student_dataset.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"\n      Dataset saved to: {dataset_path}")

    print("\n[2/3] Training models…\n")
    metrics = train_all_models(df, verbose=True)

    print("\n[3/3] Summary")
    print("-" * 40)
    print(f"Best grade classifier : {metrics['best_clf_name']}")
    print(f"  Test accuracy       : {max(v['test_accuracy'] for v in metrics['clf_results'].values()):.3f}")
    print(f"Best score regressor  : {metrics['best_reg_name']}")
    print(f"  Test R²             : {max(v['test_r2'] for v in metrics['reg_results'].values()):.3f}")
    print(f"Risk detector accuracy: {metrics['risk_accuracy']:.3f}")
    print(f"Risk detector F1      : {metrics['risk_f1']:.3f}")

    # Feature importance
    if metrics["feature_importance"]:
        print("\nTop 5 features by importance:")
        fi_sorted = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in fi_sorted[:5]:
            print(f"  {feat:<35} {imp:.4f}")

    print("\n✅ All models trained and saved to ./models/")
    print("   Run the app with:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
