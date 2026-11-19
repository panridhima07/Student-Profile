"""
Student Academic Performance Predictor
Model training, evaluation, and persistence module.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from data_generator import (
    generate_synthetic_dataset,
    FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, GRADE_ORDER
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor():
    """Return a ColumnTransformer for numeric + categorical features."""
    income_categories  = [["Low", "Medium", "High"]]
    edu_categories     = [["None", "Primary", "Secondary", "Graduate", "Postgrad"]]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=income_categories + edu_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_classifiers():
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM":                 CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42)),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6,
                                           eval_metric="mlogloss",
                                           random_state=42, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=63,
                                             random_state=42, verbose=-1)
    return models


def get_regressors():
    models = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        "Ridge Regression":  Ridge(alpha=1.0),
        "SVR":               SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6,
                                          random_state=42, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=63,
                                            random_state=42, verbose=-1)
    return models


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def encode_grade(series: pd.Series) -> np.ndarray:
    """Encode grade letters to ordinal ints (A=4, B=3, C=2, D=1, F=0)."""
    grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    return series.map(grade_map).values


def decode_grade(arr) -> list:
    grade_map = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
    return [grade_map.get(int(round(v)), "F") for v in arr]


def train_all_models(df: pd.DataFrame = None, verbose: bool = True):
    """Train classification and regression ensembles; return metrics + artifacts."""
    if df is None:
        df = generate_synthetic_dataset(n_samples=2000)

    X = df[FEATURE_COLUMNS]
    y_grade = encode_grade(df["performance_grade"])
    y_score = df["performance_score"].values
    y_risk  = df["at_risk"].values

    X_train, X_test, yg_train, yg_test, ys_train, ys_test, yr_train, yr_test = train_test_split(
        X, y_grade, y_score, y_risk, test_size=0.2, random_state=42, stratify=y_grade
    )

    preprocessor = build_preprocessor()

    # ---- Grade classification ----
    if verbose: print("Training grade classifiers…")
    clf_results = {}
    best_clf_score = -1
    best_clf_name  = None
    best_clf_pipe  = None

    for name, clf in get_classifiers().items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, yg_train, cv=cv, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, yg_train)
        preds = pipe.predict(X_test)
        acc   = accuracy_score(yg_test, preds)
        f1    = f1_score(yg_test, preds, average="weighted")
        clf_results[name] = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std":  float(np.std(cv_scores)),
            "test_accuracy": float(acc),
            "test_f1":       float(f1),
            "confusion_matrix": confusion_matrix(yg_test, preds).tolist(),
            "classification_report": classification_report(yg_test, preds,
                                        target_names=["F","D","C","B","A"], output_dict=True),
        }
        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_name  = name
            best_clf_pipe  = pipe
        if verbose: print(f"  {name}: acc={acc:.3f}, f1={f1:.3f}, cv={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")

    # ---- Score regression ----
    if verbose: print("Training score regressors…")
    reg_results = {}
    best_reg_r2   = -999
    best_reg_name = None
    best_reg_pipe = None

    for name, reg in get_regressors().items():
        pipe = Pipeline([("pre", preprocessor), ("reg", reg)])
        cv_scores = cross_val_score(pipe, X_train, ys_train, cv=5, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, ys_train)
        preds = pipe.predict(X_test)
        mae   = mean_absolute_error(ys_test, preds)
        rmse  = np.sqrt(mean_squared_error(ys_test, preds))
        r2    = r2_score(ys_test, preds)
        reg_results[name] = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std":  float(np.std(cv_scores)),
            "test_mae":  float(mae),
            "test_rmse": float(rmse),
            "test_r2":   float(r2),
        }
        if r2 > best_reg_r2:
            best_reg_r2   = r2
            best_reg_name = name
            best_reg_pipe = pipe
        if verbose: print(f"  {name}: R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

    # ---- Risk detection (binary) ----
    if verbose: print("Training risk detector…")
    from sklearn.ensemble import RandomForestClassifier as RFC
    risk_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RFC(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42))
    ])
    risk_pipe.fit(X_train, yr_train)
    risk_preds = risk_pipe.predict(X_test)
    risk_acc   = accuracy_score(yr_test, risk_preds)
    risk_f1    = f1_score(yr_test, risk_preds, average="binary")
    if verbose: print(f"  Risk detector: acc={risk_acc:.3f}, f1={risk_f1:.3f}")

    # ---- Feature importance (from best classifier) ----
    feature_importance = _extract_feature_importance(best_clf_pipe)

    # ---- Save models ----
    joblib.dump(best_clf_pipe, os.path.join(MODEL_DIR, "grade_classifier.pkl"))
    joblib.dump(best_reg_pipe, os.path.join(MODEL_DIR, "score_regressor.pkl"))
    joblib.dump(risk_pipe,     os.path.join(MODEL_DIR, "risk_detector.pkl"))
    if verbose: print(f"\nModels saved to '{MODEL_DIR}/'")
    if verbose: print(f"Best grade classifier : {best_clf_name} (acc={best_clf_score:.3f})")
    if verbose: print(f"Best score regressor  : {best_reg_name} (R²={best_reg_r2:.3f})")

    return {
        "clf_results":        clf_results,
        "reg_results":        reg_results,
        "risk_accuracy":      risk_acc,
        "risk_f1":            risk_f1,
        "best_clf_name":      best_clf_name,
        "best_reg_name":      best_reg_name,
        "feature_importance": feature_importance,
        "X_test":             X_test,
        "yg_test":            yg_test,
        "ys_test":            ys_test,
    }


def _extract_feature_importance(pipeline) -> dict:
    """Extract feature importances from the final estimator if available."""
    model = pipeline.named_steps.get("clf") or pipeline.named_steps.get("reg")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        all_features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
        return dict(zip(all_features, importances.tolist()))
    return {}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_models():
    """Load trained models from disk."""
    paths = {
        "grade": os.path.join(MODEL_DIR, "grade_classifier.pkl"),
        "score": os.path.join(MODEL_DIR, "score_regressor.pkl"),
        "risk":  os.path.join(MODEL_DIR, "risk_detector.pkl"),
    }
    models = {}
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            models[key] = None
    return models


def predict_student(student_data: dict, models: dict) -> dict:
    """
    Given a dict of student feature values, return predictions.

    Returns:
        grade          : predicted letter grade
        score          : predicted performance score
        at_risk        : bool
        risk_proba     : probability of being at risk
        grade_probas   : dict of grade → probability
        suggestions    : list of improvement suggestions
        percentile_est : estimated score percentile
    """
    df_in = pd.DataFrame([student_data])[FEATURE_COLUMNS]

    result = {}

    if models.get("grade"):
        grade_enc  = models["grade"].predict(df_in)[0]
        grade_map  = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
        result["grade"] = grade_map.get(int(round(grade_enc)), "F")
        if hasattr(models["grade"].named_steps["clf"], "predict_proba"):
            probas = models["grade"].predict_proba(df_in)[0]
            # classes are ordered by the encoder
            classes = models["grade"].named_steps["clf"].classes_
            gp = {grade_map[int(c)]: float(p) for c, p in zip(classes, probas)}
            result["grade_probas"] = gp
        else:
            result["grade_probas"] = {}

    if models.get("score"):
        result["score"] = float(np.clip(models["score"].predict(df_in)[0], 0, 100))
    else:
        result["score"] = 0.0

    if models.get("risk"):
        risk_pred  = models["risk"].predict(df_in)[0]
        result["at_risk"] = bool(risk_pred)
        if hasattr(models["risk"].named_steps["clf"], "predict_proba"):
            risk_proba = models["risk"].predict_proba(df_in)[0]
            result["risk_proba"] = float(risk_proba[1])
        else:
            result["risk_proba"] = float(risk_pred)
    else:
        result["at_risk"]     = False
        result["risk_proba"]  = 0.0

    result["suggestions"]    = generate_suggestions(student_data, result)
    result["percentile_est"] = _estimate_percentile(result["score"])
    return result


def _estimate_percentile(score: float) -> float:
    """Rough percentile based on a normal approximation (mean≈67, std≈15)."""
    from scipy.stats import norm
    return float(np.round(norm.cdf(score, loc=67, scale=15) * 100, 1))


def generate_suggestions(student_data: dict, predictions: dict) -> list:
    """Generate personalised improvement suggestions based on student profile."""
    suggestions = []
    grade = predictions.get("grade", "C")
    score = predictions.get("score", 50)

    sh  = student_data.get("study_hours_per_day", 0)
    att = student_data.get("attendance_pct", 0)
    asg = student_data.get("assignment_completion_pct", 0)
    slp = student_data.get("sleep_hours", 0)
    mh  = student_data.get("mental_health_score", 5)
    lib = student_data.get("library_visits_per_week", 0)
    ec  = student_data.get("extracurricular", 0)
    ptj = student_data.get("part_time_job", 0)
    tt  = student_data.get("travel_time_hours", 0)
    oc  = student_data.get("online_courses", 0)

    if sh < 3:
        suggestions.append("📚 Increase daily study time to at least 3–4 hours for meaningful improvement.")
    elif sh < 5 and grade in ["C", "D", "F"]:
        suggestions.append("📚 Aim for 5+ study hours/day — structured study sessions with Pomodoro can help.")

    if att < 75:
        suggestions.append("🏫 Your attendance is below 75%. Regular class attendance is strongly linked to better grades.")
    elif att < 85:
        suggestions.append("🏫 Improving attendance above 85% will give you a significant performance boost.")

    if asg < 70:
        suggestions.append("📝 Assignment completion is low. Try breaking assignments into smaller daily tasks.")
    elif asg < 90:
        suggestions.append("📝 Aim to complete all assignments — even partial submissions boost your score.")

    if slp < 6:
        suggestions.append("😴 You are sleeping fewer than 6 hours. Poor sleep impairs memory consolidation — aim for 7–8 hrs.")
    elif slp > 9:
        suggestions.append("😴 Sleeping more than 9 hours may reduce productive study time. Consider a balanced schedule.")

    if mh < 5:
        suggestions.append("🧠 Your mental health score is low. Consider speaking with a counselor or joining a support group.")
    elif mh < 7:
        suggestions.append("🧠 Mindfulness, exercise, and social activities can improve focus and mental wellbeing.")

    if lib < 1:
        suggestions.append("📖 Visit the library at least once a week — access to reference materials and a quiet environment helps.")

    if not ec:
        suggestions.append("🎯 Joining at least one extracurricular activity improves soft skills and mental health.")

    if ptj and score < 60:
        suggestions.append("⚠️ Your part-time job may be limiting study time. Consider reducing hours during exam periods.")

    if tt > 1.5:
        suggestions.append("🚌 Long commute times eat into study hours. Use commute time for revision (audio lectures, flashcards).")

    if oc == 0 and grade in ["C", "D", "F"]:
        suggestions.append("💻 Enroll in a free online course (Coursera, edX) for extra subject practice and self-paced learning.")

    if grade == "A":
        suggestions.append("🌟 Excellent! Maintain your current habits and consider mentoring peers or joining research projects.")
    elif grade == "B":
        suggestions.append("👍 Good performance! Focus on weaker subjects and aim for deeper conceptual understanding.")
    elif not suggestions:
        suggestions.append("📊 Keep improving consistently — small daily habits lead to big academic gains over time.")

    return suggestions
