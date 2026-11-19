# 🎓 Student Academic Performance Predictor

A full-stack **Machine Learning** application that predicts student academic performance, detects at-risk students, and provides personalized improvement recommendations — all through an interactive **Streamlit** dashboard.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Dashboard Tabs](#dashboard-tabs)
- [Input Features](#input-features)
- [ML Models](#ml-models)
- [Batch Prediction](#batch-prediction)
- [PDF Report Generation](#pdf-report-generation)

---

## 🎯 Project Overview

This system processes 17 student attributes — covering academic habits, lifestyle, wellbeing, and socioeconomic background — and uses an **ensemble of ML models** to:

- **Predict letter grade** (A / B / C / D / F) with probability distribution
- **Predict performance score** (0–100 continuous)
- **Detect at-risk students** who may need intervention
- **Explain predictions** using SHAP values
- **Recommend personalized actions** to improve outcomes
- **Visualize** performance via interactive Plotly dashboards

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 Real-time prediction | Instant grade + score prediction as you adjust sliders |
| 📡 Radar chart profile | Visual overview of student's strengths and weaknesses |
| 📊 Interactive EDA | Correlation heatmaps, scatter plots, box plots |
| 🤖 Model comparison | Side-by-side accuracy/R² of 6 ML algorithms |
| 🔍 SHAP explanations | Per-student feature importance (explainable AI) |
| ⚠️ Risk detection | Binary classifier to flag at-risk students |
| 📋 Personalised tips | Rule-based suggestions tailored to each student |
| 📜 History tracking | Log and compare multiple predictions per session |
| 📥 Export | Download predictions as CSV or Excel |
| 📄 PDF report | Generate a formatted PDF report for each student |

---

## 🛠 Tech Stack

| Layer | Library / Tool |
|---|---|
| **Dashboard** | Streamlit 1.67 |
| **Visualizations** | Plotly, Seaborn, Matplotlib |
| **ML — Classification** | Random Forest, XGBoost, LightGBM, GradientBoosting, Logistic Regression, SVM |
| **ML — Regression** | Random Forest, XGBoost, LightGBM, GradientBoosting, Ridge, SVR |
| **XAI** | SHAP |
| **Data** | Pandas, NumPy, Scikit-learn |
| **Report** | fpdf2, openpyxl |
| **Serialization** | Joblib |

---

## 📁 Project Structure

```
student profile/
├── app.py                  # Main Streamlit dashboard
├── data_generator.py       # Synthetic dataset creation & feature definitions
├── model_trainer.py        # Model training, evaluation & inference
├── train_models.py         # Standalone training script (run once)
├── batch_predict.py        # CLI batch prediction for CSV files
├── report_generator.py     # PDF report generation
├── run_app.bat             # Windows one-click launcher
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Saved trained models (auto-created)
│   ├── grade_classifier.pkl
│   ├── score_regressor.pkl
│   └── risk_detector.pkl
├── data/                   # Generated datasets (auto-created)
│   └── student_dataset.csv
└── reports/                # Generated PDF reports (auto-created)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install scikit-learn xgboost lightgbm streamlit plotly seaborn matplotlib shap joblib imbalanced-learn scipy statsmodels fpdf2 openpyxl
```

---

## ▶ Running the App

### Option 1 — Double-click launcher (Windows)
```
run_app.bat
```
This automatically trains models if not present, then opens the dashboard.

### Option 2 — Manual

**Step 1:** Train models (only needed once)
```bash
python train_models.py
```

**Step 2:** Launch dashboard
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🖥 Dashboard Tabs

### 🔮 Predict
- Adjust sliders in the sidebar to set student profile
- Click **Predict Performance** to get instant results
- View: grade badge, score gauge, risk assessment, radar chart, personalised suggestions

### 📊 EDA & Analytics
- Grade & score distributions
- Feature vs performance scatter plots (with OLS trendline)
- Correlation heatmap
- Box plots by category (grade, income, parental education…)

### 🤖 Model Performance
- Accuracy, F1, R², MAE, RMSE for all 6 algorithms
- Cross-validation scores
- Highlighted best model

### 📈 Feature Insights
- Feature importance bar chart (from best classifier)
- SHAP force values for current student input
- Top-5 feature scatter matrix

### 📜 History & Export
- Track all predictions made in the session
- Score trend line across predictions
- Download as CSV or Excel

---

## 📝 Input Features (17 total)

| Feature | Type | Range |
|---|---|---|
| Daily Study Hours | Numeric | 1–12 hrs |
| Attendance % | Numeric | 40–100% |
| Previous Semester Marks | Numeric | 30–100 |
| Assignment Completion % | Numeric | 30–100% |
| Number of Subjects | Integer | 3–8 |
| Sleep Hours/Night | Numeric | 4–10 hrs |
| Extracurricular Activities | Binary | Yes/No |
| Internet Access at Home | Binary | Yes/No |
| Family Income Level | Categorical | Low/Medium/High |
| Parental Education | Categorical | None/Primary/Secondary/Graduate/Postgrad |
| Mental Health Score | Numeric | 1–10 |
| Peer Group Quality | Numeric | 1–10 |
| Teacher Quality | Numeric | 1–10 |
| Library Visits/Week | Integer | 0–7 |
| Online Courses Enrolled | Integer | 0–5 |
| Part-Time Job | Binary | Yes/No |
| Daily Travel Time | Numeric | 0–3 hrs |

---

## 🤖 ML Models

### Grade Classifier (5-class: A/B/C/D/F)
Models trained with 5-fold stratified cross-validation:
- Random Forest (200 estimators)
- XGBoost (200 estimators, lr=0.08)
- LightGBM (200 estimators, num_leaves=63)
- Gradient Boosting
- Logistic Regression
- SVM (RBF kernel)

### Score Regressor (continuous 0–100)
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gradient Boosting Regressor
- Ridge Regression
- SVR (RBF)

### At-Risk Binary Classifier
- Random Forest with class_weight="balanced"
- Flags students with predicted score < 50

---

## 📦 Batch Prediction

Predict for multiple students from a CSV file:

```bash
python batch_predict.py --input my_students.csv --output results.csv
```

The input CSV should include the same column names as the feature list above.
Missing columns are filled with sensible defaults automatically.

---

## 📄 PDF Report

Generate a PDF report programmatically:

```python
from report_generator import generate_pdf_report
path = generate_pdf_report(student_data_dict, prediction_result_dict)
print(f"Report saved to: {path}")
```

---

## 📌 Notes

- The dataset is **synthetic** (2,000 generated records) designed to mirror real-world patterns.
- Models are saved to `./models/` as `.pkl` files and loaded on subsequent runs.
- The app auto-trains on first launch if no saved models are found.
- SHAP explanations use a background of 100 samples for speed.

---

*Built with ❤️ using Python, Streamlit, scikit-learn, XGBoost, LightGBM & SHAP*
