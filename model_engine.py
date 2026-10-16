"""CSV-backed training and inference utilities for EduPredict."""
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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


def _normalise_dataframe(dataframe):
    """Accept the specified CSV headings and common older EduPredict headings."""
    aliases = {
        "attendance_rate": "attendance_percentage",
        "lms_activity_score": "lms_logins_last_30_days",
        "gpa": "previous_gpa",
    }
    dataframe = dataframe.rename(columns={old: new for old, new in aliases.items() if old in dataframe.columns and new not in dataframe.columns})
    for column in FEATURES:
        if column not in dataframe.columns:
            dataframe[column] = 0.0
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)
    return dataframe


def train_model(csv_path=CSV_PATH):
    """Train from EduPredict.csv and save a model plus its fitted feature scaler."""
    dataframe = _normalise_dataframe(pd.read_csv(csv_path))
    # Prefer an outcome measured after the midterm, leaving midterm_score as an
    # input signal. Smaller datasets that only have a midterm score still work.
    target_column = next((column for column in ("final_score", "performance_score", "final_grade") if column in dataframe.columns), "midterm_score")
    target = pd.to_numeric(dataframe[target_column], errors="coerce").fillna(0.0)
    # A dataset with a constant target cannot form a meaningful prediction model.
    if target.nunique() < 2:
        raise ValueError("EduPredict.csv needs at least two distinct midterm_score values to train the model.")
    scaler = StandardScaler().fit(dataframe[FEATURES])
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(scaler.transform(dataframe[FEATURES]), target)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    return model, scaler


def _load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        if not os.path.exists(CSV_PATH):
            return None, None
        return train_model()
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


if __name__ == "__main__":
    train_model()
    print(f"Model saved to {MODEL_PATH}")
