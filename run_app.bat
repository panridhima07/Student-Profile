@echo off
REM ============================================================
REM  Student Academic Performance Predictor — Launcher
REM ============================================================
echo.
echo ============================================================
echo   Student Academic Performance Predictor
echo ============================================================
echo.

SET PYTHON=C:\Users\ridhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
SET STREAMLIT=%PYTHON:python.exe=Scripts\streamlit.exe%
SET PROJECT_DIR=%~dp0

REM Check if models exist, train if not
IF NOT EXIST "%PROJECT_DIR%models\grade_classifier.pkl" (
    echo [1/2] Training ML models for the first time...
    echo       This will take about 30-60 seconds.
    echo.
    "%PYTHON%" "%PROJECT_DIR%train_models.py"
    echo.
)

echo [2/2] Launching Streamlit dashboard...
echo.
echo       Open your browser at: http://localhost:8501
echo.
"%STREAMLIT%" run "%PROJECT_DIR%app.py" --server.port 8501 --server.headless false

pause
