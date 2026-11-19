"""
report_generator.py — Generate a PDF performance report for a student.

Usage:
    from report_generator import generate_pdf_report
    path = generate_pdf_report(student_data, prediction_result)
"""

import os
from fpdf import FPDF
import pandas as pd
from datetime import datetime

GRADE_COLORS_HEX = {
    "A": (46, 204, 113),
    "B": (52, 152, 219),
    "C": (243, 156, 18),
    "D": (230, 126, 34),
    "F": (231, 76, 60),
}


class StudentReportPDF(FPDF):
    def header(self):
        self.set_fill_color(102, 126, 234)
        self.rect(0, 0, 210, 30, "F")
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 14, "Student Academic Performance Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Student Performance Predictor", align="C")

    def section_title(self, title: str):
        self.set_fill_color(240, 243, 255)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def kv_row(self, key: str, value: str, highlight: bool = False):
        if highlight:
            self.set_fill_color(255, 253, 230)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(85, 7, f"  {key}", fill=True, border="B")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        self.cell(0, 7, f"  {value}", fill=highlight, border="B", new_x="LMARGIN", new_y="NEXT")

    def grade_badge(self, grade: str, score: float):
        r, g, b = GRADE_COLORS_HEX.get(grade, (127, 140, 141))
        self.set_fill_color(r, g, b)
        self.set_font("Helvetica", "B", 36)
        self.set_text_color(255, 255, 255)
        x = self.get_x()
        self.cell(50, 20, grade, fill=True, align="C")
        self.set_x(x + 55)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 20, f"Score: {score:.1f} / 100", align="L", new_x="LMARGIN", new_y="NEXT")

    def suggestion_item(self, text: str, idx: int):
        self.set_fill_color(248, 249, 250)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        self.multi_cell(0, 7, f"  {idx}. {text}", fill=True, border="L")
        self.ln(1)


def generate_pdf_report(student_data: dict, result: dict, output_path: str = None) -> str:
    """Generate a PDF report and return its file path."""
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), "reports", f"student_report_{ts}.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = StudentReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Prediction Summary ───────────────────────────────────────────────────
    pdf.section_title("Prediction Summary")
    pdf.grade_badge(result.get("grade", "?"), result.get("score", 0))
    pdf.ln(4)

    risk_text = "AT RISK ⚠️" if result.get("at_risk") else "On Track ✅"
    pdf.kv_row("Risk Status",             risk_text,                             highlight=result.get("at_risk", False))
    pdf.kv_row("Risk Probability",        f"{result.get('risk_proba', 0)*100:.1f}%")
    pdf.kv_row("Estimated Percentile",    f"{result.get('percentile_est', 0):.1f}th")
    pdf.ln(6)

    # ── Student Profile ──────────────────────────────────────────────────────
    pdf.section_title("Student Input Profile")
    labels = {
        "study_hours_per_day":       "Daily Study Hours",
        "attendance_pct":            "Attendance %",
        "prev_semester_marks":       "Previous Semester Marks",
        "assignment_completion_pct": "Assignment Completion %",
        "num_subjects":              "Number of Subjects",
        "sleep_hours":               "Sleep Hours/Night",
        "extracurricular":           "Extracurricular Activities",
        "internet_access":           "Internet Access",
        "family_income_level":       "Family Income Level",
        "parental_education":        "Parental Education",
        "mental_health_score":       "Mental Health Score",
        "peer_group_quality":        "Peer Group Quality",
        "teacher_quality":           "Teacher Quality",
        "library_visits_per_week":   "Library Visits/Week",
        "online_courses":            "Online Courses",
        "part_time_job":             "Part-Time Job",
        "travel_time_hours":         "Travel Time (hrs/day)",
    }
    bool_fields = {"extracurricular", "internet_access", "part_time_job"}
    for key, label in labels.items():
        val = student_data.get(key, "N/A")
        if key in bool_fields:
            val = "Yes" if val else "No"
        pdf.kv_row(label, str(val))
    pdf.ln(6)

    # ── Suggestions ──────────────────────────────────────────────────────────
    pdf.section_title("Personalised Improvement Suggestions")
    suggestions = result.get("suggestions", [])
    for i, s in enumerate(suggestions, 1):
        # Strip emoji for PDF (basic latin fallback)
        clean_s = s.encode("latin-1", errors="replace").decode("latin-1")
        pdf.suggestion_item(clean_s, i)
    pdf.ln(6)

    # ── Grade Probabilities ───────────────────────────────────────────────────
    gp = result.get("grade_probas", {})
    if gp:
        pdf.section_title("Grade Probability Breakdown")
        for grade in ["A", "B", "C", "D", "F"]:
            prob = gp.get(grade, 0)
            bar_width = int(prob * 150)
            r, g, b = GRADE_COLORS_HEX.get(grade, (127, 140, 141))
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(12, 8, grade)
            pdf.set_fill_color(r, g, b)
            pdf.cell(bar_width if bar_width > 0 else 1, 8, "", fill=True)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(150 - bar_width, 8, "", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(30, 8, f"  {prob*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    pdf.output(output_path)
    return output_path
