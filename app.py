"""
Student Academic Performance Predictor — Streamlit Dashboard
Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import (
    generate_synthetic_dataset, FEATURE_COLUMNS, GRADE_COLORS,
    GRADE_ORDER, FEATURE_DESCRIPTIONS
)
from model_trainer import (
    train_all_models, load_models, predict_student
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main {background-color: #f8f9fa;}

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label {color: #666; font-size: 13px;}
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 2rem; font-weight: 700;
    }

    /* Grade badge */
    .grade-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 2.4rem;
        font-weight: 900;
        color: white;
        text-align: center;
    }

    /* Suggestion cards */
    .suggestion-card {
        background: #fff;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }
    .risk-card {
        background: #fff5f5;
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .safe-card {
        background: #f0fff4;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 4px;
        border-bottom: 2px solid #667eea;
    }

    /* Tab tweaks */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);}
    section[data-testid="stSidebar"] * {color: #ecf0f1 !important;}
    section[data-testid="stSidebar"] .stSlider label {color: #bdc3c7 !important;}
    section[data-testid="stSidebar"] .stSelectbox label {color: #bdc3c7 !important;}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "models_trained": False,
        "models":         None,
        "train_metrics":  None,
        "dataset":        None,
        "history":        [],       # list of past predictions
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.markdown("---")

    st.markdown("### 📚 Academic Inputs")
    study_hours  = st.slider("Daily Study Hours",        1.0, 12.0, 4.0, 0.5)
    attendance   = st.slider("Attendance %",            40.0, 100.0, 80.0, 1.0)
    prev_marks   = st.slider("Previous Semester Marks", 30.0, 100.0, 65.0, 1.0)
    assignment   = st.slider("Assignment Completion %", 30.0, 100.0, 75.0, 1.0)
    num_subjects = st.slider("Number of Subjects",          3,     8,    5,    1)

    st.markdown("### 🌙 Lifestyle")
    sleep_hours  = st.slider("Sleep Hours / Night", 4.0, 10.0, 7.0, 0.5)
    travel_time  = st.slider("Travel Time (hrs/day)", 0.0, 3.0, 0.5, 0.25)
    part_time_job = st.selectbox("Has Part-Time Job", ["No", "Yes"])

    st.markdown("### 🧠 Wellbeing & Environment")
    mental_health = st.slider("Mental Health Score (1–10)",  1.0, 10.0, 7.0, 0.5)
    peer_quality  = st.slider("Peer Group Quality (1–10)",   1.0, 10.0, 6.0, 0.5)
    teacher_qual  = st.slider("Teacher Quality (1–10)",      1.0, 10.0, 7.0, 0.5)

    st.markdown("### 📖 Resources & Activities")
    library_visits  = st.slider("Library Visits / Week",  0, 7, 2)
    online_courses  = st.slider("Online Courses Enrolled", 0, 5, 1)
    extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    internet_access = st.selectbox("Internet Access at Home",    ["No", "Yes"])

    st.markdown("### 👨‍👩‍👦 Background")
    family_income   = st.selectbox("Family Income Level",   ["Low", "Medium", "High"])
    parental_edu    = st.selectbox("Parental Education", ["None", "Primary", "Secondary", "Graduate", "Postgrad"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Performance", use_container_width=True, type="primary")
    train_btn   = st.button("🏋️ (Re)Train Models",    use_container_width=True)


# ── Collect input ─────────────────────────────────────────────────────────────
student_input = {
    "study_hours_per_day":       study_hours,
    "attendance_pct":            attendance,
    "prev_semester_marks":       prev_marks,
    "assignment_completion_pct": assignment,
    "num_subjects":              num_subjects,
    "sleep_hours":               sleep_hours,
    "extracurricular":           1 if extracurricular == "Yes" else 0,
    "internet_access":           1 if internet_access == "Yes" else 0,
    "family_income_level":       family_income,
    "parental_education":        parental_edu,
    "mental_health_score":       mental_health,
    "peer_group_quality":        peer_quality,
    "teacher_quality":           teacher_qual,
    "library_visits_per_week":   library_visits,
    "online_courses":            online_courses,
    "part_time_job":             1 if part_time_job == "Yes" else 0,
    "travel_time_hours":         travel_time,
}


# ── Auto-load or train models ─────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner=False)
def _load_or_train():
    grade_path = os.path.join(MODEL_DIR, "grade_classifier.pkl")
    if os.path.exists(grade_path):
        return load_models(), None, generate_synthetic_dataset()
    else:
        df = generate_synthetic_dataset()
        metrics = train_all_models(df, verbose=False)
        return load_models(), metrics, df

if train_btn:
    st.cache_resource.clear()
    with st.spinner("🏋️ Training models on synthetic dataset — this takes ~30s…"):
        df = generate_synthetic_dataset()
        metrics = train_all_models(df, verbose=False)
        models  = load_models()
        st.session_state["models"]         = models
        st.session_state["train_metrics"]  = metrics
        st.session_state["dataset"]        = df
        st.session_state["models_trained"] = True
    st.success("✅ Models trained and saved!")

if not st.session_state["models_trained"]:
    with st.spinner("⚙️ Loading / training models on first run…"):
        models, metrics, df = _load_or_train()
        st.session_state["models"]         = models
        st.session_state["train_metrics"]  = metrics
        st.session_state["dataset"]        = df
        st.session_state["models_trained"] = True


models  = st.session_state["models"]
metrics = st.session_state["train_metrics"]
df      = st.session_state["dataset"]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;">
    <h1 style="color:white; margin:0; font-size:2.2rem;">🎓 Student Academic Performance Predictor</h1>
    <p style="color: rgba(255,255,255,0.85); margin:6px 0 0 0; font-size:1.05rem;">
        ML-powered insights to predict grades, detect risk & personalise learning advice
    </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict",
    "📊 EDA & Analytics",
    "🤖 Model Performance",
    "📈 Feature Insights",
    "📜 History & Export",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn or True:   # show results panel always (use last prediction)
        result = predict_student(student_input, models)

        # Store in history
        if predict_btn:
            entry = {**student_input, **result, "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
            # serialise non-primitive types
            entry["suggestions"]    = result["suggestions"]
            entry["grade_probas"]   = result["grade_probas"]
            st.session_state["history"].append(entry)

        grade  = result["grade"]
        score  = result["score"]
        at_risk = result["at_risk"]
        risk_p  = result["risk_proba"]
        pct     = result["percentile_est"]
        gp      = result.get("grade_probas", {})
        sugg    = result["suggestions"]

        grade_color = GRADE_COLORS.get(grade, "#7f8c8d")

        # ── KPI Row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div style="text-align:center; background:white; border-radius:16px;
                        padding:20px; box-shadow:0 2px 12px rgba(0,0,0,0.08);">
                <div style="font-size:0.85rem; color:#888; margin-bottom:4px;">PREDICTED GRADE</div>
                <div class="grade-badge" style="background:{grade_color};">{grade}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.metric("🎯 Performance Score", f"{score:.1f} / 100")
        with c3:
            st.metric("📈 Estimated Percentile", f"{pct}th")
        with c4:
            risk_color = "#e74c3c" if at_risk else "#2ecc71"
            risk_text  = "AT RISK ⚠️"  if at_risk else "ON TRACK ✅"
            st.metric("🚦 Risk Status", risk_text)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Grade probability chart + Gauge ──────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Grade Probability Distribution</div>', unsafe_allow_html=True)
            if gp:
                labels = list(gp.keys())
                vals   = [gp.get(g, 0) for g in labels]
                colors = [GRADE_COLORS.get(g, "#95a5a6") for g in labels]
                fig = go.Figure(go.Bar(
                    x=labels, y=[v*100 for v in vals],
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in vals],
                    textposition="outside",
                ))
                fig.update_layout(
                    yaxis_title="Probability (%)", xaxis_title="Grade",
                    yaxis_range=[0, 110], plot_bgcolor="white",
                    paper_bgcolor="white", height=300,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Performance Score Gauge</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": 60},
                title={"text": "Score (0–100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": grade_color, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  40],  "color": "#fadbd8"},
                        {"range": [40, 55],  "color": "#fdebd0"},
                        {"range": [55, 70],  "color": "#fef9e7"},
                        {"range": [70, 85],  "color": "#eafaf1"},
                        {"range": [85, 100], "color": "#d5f5e3"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 60},
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # ── Risk panel ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
        if at_risk:
            st.markdown(f"""
            <div class="risk-card">
                ⚠️ <strong>This student is at risk of poor academic performance.</strong><br>
                Risk probability: <strong>{risk_p*100:.1f}%</strong><br>
                Immediate intervention is recommended.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-card">
                ✅ <strong>This student is on track.</strong><br>
                Risk probability: <strong>{risk_p*100:.1f}%</strong><br>
                Keep up the good habits and continue improving.
            </div>""", unsafe_allow_html=True)

        # ── Suggestions ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Personalised Improvement Suggestions</div>', unsafe_allow_html=True)
        for s in sugg:
            st.markdown(f'<div class="suggestion-card">{s}</div>', unsafe_allow_html=True)

        # ── Radar chart ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📡 Student Profile Radar</div>', unsafe_allow_html=True)
        radar_features = ["study_hours_per_day", "attendance_pct", "prev_semester_marks",
                          "assignment_completion_pct", "sleep_hours", "mental_health_score",
                          "peer_group_quality", "teacher_quality"]
        radar_labels   = ["Study hrs", "Attendance", "Prev Marks", "Assignments",
                          "Sleep", "Mental Health", "Peer Quality", "Teacher Quality"]
        radar_max      = [12, 100, 100, 100, 10, 10, 10, 10]
        radar_vals     = [student_input.get(f, 0) / m * 10 for f, m in zip(radar_features, radar_max)]
        radar_vals_pct = [(v / 10) * 100 for v in radar_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_vals_pct,
            theta=radar_labels,
            fill='toself',
            fillcolor=f"rgba(102, 126, 234, 0.25)",
            line_color="#667eea",
            name="Your Profile"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=400,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown(f"Dataset: **{len(df):,} students**, **{len(FEATURE_COLUMNS)} features**")

    # Grade distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Grade Distribution</div>', unsafe_allow_html=True)
        grade_counts = df["performance_grade"].value_counts().reindex(GRADE_ORDER).fillna(0)
        fig = px.bar(
            x=grade_counts.index, y=grade_counts.values,
            color=grade_counts.index,
            color_discrete_map=GRADE_COLORS,
            labels={"x": "Grade", "y": "Count"},
            text=grade_counts.values,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="performance_score", nbins=40,
            color_discrete_sequence=["#667eea"],
            labels={"performance_score": "Performance Score"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Feature vs score scatter
    st.markdown('<div class="section-header">Feature vs Performance Score</div>', unsafe_allow_html=True)
    num_feat = st.selectbox(
        "Select numeric feature",
        options=[f for f in FEATURE_COLUMNS if f not in ["family_income_level", "parental_education"]],
        format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
        key="eda_feat"
    )
    fig = px.scatter(
        df, x=num_feat, y="performance_score",
        color="performance_grade",
        color_discrete_map=GRADE_COLORS,
        opacity=0.5,
        trendline="ols",
        labels={num_feat: FEATURE_DESCRIPTIONS.get(num_feat, num_feat), "performance_score": "Score"},
        height=400,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
    numeric_df = df[
        [c for c in FEATURE_COLUMNS if c not in ["family_income_level","parental_education"]]
        + ["performance_score"]
    ].copy()
    corr = numeric_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels={"color": "Correlation"},
        height=550,
    )
    fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Box plots
    st.markdown('<div class="section-header">Score by Category</div>', unsafe_allow_html=True)
    cat_col = st.selectbox("Categorical feature", ["performance_grade", "family_income_level",
                                                    "parental_education", "extracurricular", "part_time_job"],
                            key="cat_box")
    fig = px.box(
        df, x=cat_col, y="performance_score",
        color=cat_col,
        points="outliers",
        labels={"performance_score": "Score"},
        height=380,
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # At-risk stats
    st.markdown('<div class="section-header">At-Risk Student Statistics</div>', unsafe_allow_html=True)
    risk_df = df.groupby("at_risk")["performance_score"].describe()
    risk_df.index = ["On Track", "At Risk"]
    st.dataframe(risk_df.style.format("{:.2f}"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Model Performance Dashboard")

    if metrics:
        col1, col2 = st.columns(2)

        # ---- Classifier comparison ----
        with col1:
            st.markdown('<div class="section-header">Grade Classifier Comparison</div>', unsafe_allow_html=True)
            clf_data = pd.DataFrame([
                {
                    "Model": name,
                    "CV Accuracy": f"{v['cv_mean']:.3f} ± {v['cv_std']:.3f}",
                    "Test Accuracy": v["test_accuracy"],
                    "F1 Score": v["test_f1"],
                }
                for name, v in metrics["clf_results"].items()
            ])
            best_clf = metrics["best_clf_name"]
            styled = clf_data.style.apply(
                lambda row: ["background: #eafaf1; font-weight:bold" if row["Model"] == best_clf else "" for _ in row],
                axis=1
            ).format({"Test Accuracy": "{:.3f}", "F1 Score": "{:.3f}"})
            st.dataframe(styled, use_container_width=True)

            # Accuracy bar chart
            fig = px.bar(
                clf_data, x="Model", y="Test Accuracy",
                color="Test Accuracy",
                color_continuous_scale="Greens",
                text=[f"{v:.3f}" for v in clf_data["Test Accuracy"]],
                height=320,
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ---- Regressor comparison ----
        with col2:
            st.markdown('<div class="section-header">Score Regressor Comparison</div>', unsafe_allow_html=True)
            reg_data = pd.DataFrame([
                {
                    "Model":    name,
                    "CV R²":    f"{v['cv_r2_mean']:.3f} ± {v['cv_r2_std']:.3f}",
                    "Test R²":  v["test_r2"],
                    "MAE":      v["test_mae"],
                    "RMSE":     v["test_rmse"],
                }
                for name, v in metrics["reg_results"].items()
            ])
            best_reg = metrics["best_reg_name"]
            styled2 = reg_data.style.apply(
                lambda row: ["background: #eaf2ff; font-weight:bold" if row["Model"] == best_reg else "" for _ in row],
                axis=1
            ).format({"Test R²": "{:.3f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"})
            st.dataframe(styled2, use_container_width=True)

            fig = px.bar(
                reg_data, x="Model", y="Test R²",
                color="Test R²",
                color_continuous_scale="Blues",
                text=[f"{v:.3f}" for v in reg_data["Test R²"]],
                height=320,
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ---- Risk detector metrics ----
        st.markdown('<div class="section-header">At-Risk Detector (Binary Classifier)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['risk_accuracy']:.3f}")
        c2.metric("F1 Score (At-Risk)", f"{metrics['risk_f1']:.3f}")

    else:
        st.info("Train the models first using the sidebar button to see performance metrics.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE INSIGHTS (SHAP / IMPORTANCE)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📈 Feature Importance & Insights")

    if metrics and metrics.get("feature_importance"):
        fi = metrics["feature_importance"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values(
            "Importance", ascending=True
        )
        fi_df["Feature Label"] = fi_df["Feature"].map(lambda x: FEATURE_DESCRIPTIONS.get(x, x))

        st.markdown('<div class="section-header">Top Feature Importances (Best Classifier)</div>', unsafe_allow_html=True)
        fig = px.bar(
            fi_df, x="Importance", y="Feature Label",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            height=500,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models to see feature importances.")

    # SHAP explanation
    st.markdown('<div class="section-header">🔍 SHAP Feature Explanation for Current Input</div>', unsafe_allow_html=True)
    try:
        import shap
        if models.get("score"):
            with st.spinner("Computing SHAP values…"):
                bg_data = df[FEATURE_COLUMNS].sample(100, random_state=42)
                pipeline = models["score"]
                pre = pipeline.named_steps["pre"]
                reg = pipeline.named_steps["reg"]
                bg_transformed = pre.transform(bg_data)
                inp_transformed = pre.transform(pd.DataFrame([student_input])[FEATURE_COLUMNS])
                all_features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

                explainer = shap.Explainer(reg, bg_transformed,
                                            feature_names=all_features)
                shap_values = explainer(inp_transformed)

                sv = shap_values.values[0]
                base = shap_values.base_values[0]
                shap_df = pd.DataFrame({
                    "Feature": [FEATURE_DESCRIPTIONS.get(f, f) for f in all_features],
                    "SHAP Value": sv,
                    "Direction": ["Positive ▲" if v > 0 else "Negative ▼" for v in sv]
                }).sort_values("SHAP Value", key=abs, ascending=False)

                fig = px.bar(
                    shap_df, x="SHAP Value", y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    text=[f"{v:+.2f}" for v in shap_df["SHAP Value"]],
                    height=460,
                    title=f"SHAP values (base score: {base:.1f})",
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Positive SHAP → feature pushes score up | Negative SHAP → feature pulls score down")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # Pairplot-style scatter matrix
    st.markdown('<div class="section-header">Top-5 Feature Scatter Matrix</div>', unsafe_allow_html=True)
    top5 = ["study_hours_per_day", "attendance_pct", "prev_semester_marks",
            "assignment_completion_pct", "sleep_hours"]
    fig = px.scatter_matrix(
        df, dimensions=top5,
        color="performance_grade",
        color_discrete_map=GRADE_COLORS,
        opacity=0.4,
        height=600,
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY & EXPORT
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📜 Prediction History & Export")

    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Go to the **🔮 Predict** tab and hit **Predict Performance**.")
    else:
        # Build a tidy dataframe for display
        display_rows = []
        for h in history:
            display_rows.append({
                "Timestamp":           h.get("timestamp", ""),
                "Study Hrs":           h.get("study_hours_per_day"),
                "Attendance %":        h.get("attendance_pct"),
                "Prev Marks":          h.get("prev_semester_marks"),
                "Assignment %":        h.get("assignment_completion_pct"),
                "Sleep Hrs":           h.get("sleep_hours"),
                "Predicted Grade":     h.get("grade", ""),
                "Predicted Score":     round(h.get("score", 0), 1),
                "At Risk":             "⚠️ Yes" if h.get("at_risk") else "✅ No",
                "Risk Proba %":        round(h.get("risk_proba", 0) * 100, 1),
                "Est. Percentile":     h.get("percentile_est"),
            })
        hist_df = pd.DataFrame(display_rows)
        st.dataframe(hist_df, use_container_width=True)

        # Score trend over sessions
        if len(history) > 1:
            st.markdown('<div class="section-header">Score Trend Over Predictions</div>', unsafe_allow_html=True)
            fig = px.line(
                hist_df, x="Timestamp", y="Predicted Score",
                markers=True, text="Predicted Grade",
                color_discrete_sequence=["#667eea"],
                height=300,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download History (CSV)",
                data=csv_data,
                file_name="student_performance_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            try:
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    hist_df.to_excel(writer, index=False, sheet_name="Predictions")
                buf.seek(0)
                st.download_button(
                    "📥 Download History (Excel)",
                    data=buf,
                    file_name="student_performance_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                pass

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state["history"] = []
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.85rem;">
    🎓 Student Performance Predictor &nbsp;|&nbsp; 
    Built with Streamlit · scikit-learn · XGBoost · LightGBM · SHAP · Plotly &nbsp;|&nbsp;
    Synthetic data for demonstration
</div>
""", unsafe_allow_html=True)
