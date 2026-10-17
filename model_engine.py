"""Runtime inference utilities for models trained in EduPredict_Model_Training.ipynb."""
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "EduPredict.csv")
MODEL_PATH = os.path.join(BASE_DIR, "edupredict_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES = ["attendance_percentage", "lms_logins_last_30_days", "previous_gpa", "midterm_score"]


def _number(record, key, legacy_key=None):
    value = record.get(key, record.get(legacy_key, 0) if legacy_key else 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    with open(MODEL_PATH, "rb") as model_file, open(SCALER_PATH, "rb") as scaler_file:
        return pickle.load(model_file), pickle.load(scaler_file)


def prediction_risk(predicted_gpa, attendance_percentage, logins):
    if predicted_gpa < 2.0 or attendance_percentage < 60 or logins < 8:
        return "High"
    if predicted_gpa < 2.7 or attendance_percentage < 75 or logins < 15:
        return "Medium"
    return "Low"


def predict_student_performance(student):
    """Return a GPA estimate, predicted performance score, and dropout risk."""
    attendance = _number(student, "attendance_percentage", "attendance_rate")
    logins = _number(student, "lms_logins_last_30_days", "lms_activity_score")
    prior_gpa = _number(student, "previous_gpa")
    midterm = _number(student, "midterm_score")
    model, scaler = _load_model()
    if model is None:
        # Keeps registration and dashboards functional before the CSV is supplied.
        score = max(0.0, min(100.0, attendance * .45 + min(logins, 30) * .7 + prior_gpa * 8))
    else:
        score = float(model.predict(scaler.transform([[attendance, logins, prior_gpa, midterm]]))[0])
    gpa = round(max(0.0, min(4.0, score / 25 if score > 4 else score)), 2)
    return {"predicted_performance": round(score, 2), "predicted_gpa": gpa, "dropout_risk": prediction_risk(gpa, attendance, logins)}


