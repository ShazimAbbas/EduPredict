"""Inference helpers for EduPredict's subject-score pass/fail model."""
import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCORE_COLUMNS = ["math_score", "history_score", "physics_score", "chemistry_score", "biology_score", "english_score", "geography_score"]
CATEGORICAL_COLUMNS = ["gender", "part_time_job", "extracurricular_activities", "career_aspiration"]
NUMERIC_COLUMNS = ["absence_days", "weekly_self_study_hours", *SCORE_COLUMNS]


def _load_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
        encoders = {key: joblib.load(os.path.join(BASE_DIR, f"{key}_encoder.pkl")) for key in CATEGORICAL_COLUMNS}
        return model, columns, encoders
    except (FileNotFoundError, ModuleNotFoundError, EOFError):
        return None, [], {}


def _value(record, key, default=0):
    value = record.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode(value, encoder):
    mapping = {str(label): index for index, label in enumerate(encoder.classes_)}
    return mapping.get(str(value), 0)


def build_input_frame(student, columns, encoders):
    """Create an encoded, ordered frame matching the saved model schema."""
    values = {
        "gender": student.get("gender", "unknown"),
        "part_time_job": str(student.get("part_time_job", False)),
        "extracurricular_activities": str(student.get("extracurricular_activities", False)),
        "career_aspiration": student.get("career_aspiration", "unknown"),
        **{column: _value(student, column) for column in NUMERIC_COLUMNS},
    }
    for column in CATEGORICAL_COLUMNS:
        values[column] = _encode(values[column], encoders[column])
    return pd.DataFrame([[values.get(column, 0) for column in columns]], columns=columns)


def improvement_suggestions(student):
    suggestions = []
    weak_subjects = [column.replace("_score", "").title() for column in SCORE_COLUMNS if _value(student, column) < 40]
    if weak_subjects:
        suggestions.append("Focus on " + ", ".join(weak_subjects[:3]) + " through targeted practice and teacher guidance.")
    if _value(student, "absence_days") > 10:
        suggestions.append("Improve attendance consistency to avoid missed lessons.")
    if _value(student, "weekly_self_study_hours") < 5:
        suggestions.append("Schedule at least five focused self-study hours each week.")
    if not suggestions:
        suggestions.append("Maintain current study habits and continue reviewing each subject weekly.")
    return " ".join(suggestions)


def predict_student_performance(student):
    """Return predicted marks, pass/fail result, risk level, and actionable feedback."""
    expected_marks = round(sum(_value(student, column) for column in SCORE_COLUMNS) / len(SCORE_COLUMNS), 1)
    model, columns, encoders = _load_artifacts()
    if model is not None:
        frame = build_input_frame(student, columns, encoders)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1])
    else:
        passed = sum(_value(student, column) >= 40 for column in SCORE_COLUMNS) >= 6
        prediction, probability = int(passed), 0.75 if passed else 0.25
    status = "Pass" if prediction else "Fail"
    risk = "Low" if probability >= .80 else "Medium" if probability >= .50 else "High"
    return {
        "expected_marks": expected_marks,
        "predicted_gpa": round(min(4, expected_marks / 25), 2),
        "pass_fail": status,
        "dropout_risk": risk,
        "confidence": round(probability, 3),
        "improvement_suggestion": improvement_suggestions(student),
    }
