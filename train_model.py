"""Train EduPredict's pass/fail model from the bundled student score dataset."""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "student_scores_dataset.csv")
CATEGORICAL_COLUMNS = ["gender", "part_time_job", "extracurricular_activities", "career_aspiration"]
SCORE_COLUMNS = ["math_score", "history_score", "physics_score", "chemistry_score", "biology_score", "english_score", "geography_score"]
ARTIFACTS = ["model.pkl", "model_columns.pkl", *[f"{column}_encoder.pkl" for column in CATEGORICAL_COLUMNS]]


def train_model():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("student_scores_dataset.csv is required to train the model.")
    dataframe = pd.read_csv(DATASET_PATH).drop(columns=["id", "first_name", "last_name", "email"], errors="ignore")
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column].astype(str))
        joblib.dump(encoder, os.path.join(BASE_DIR, f"{column}_encoder.pkl"))
    dataframe["pass_fail"] = ((dataframe[SCORE_COLUMNS] >= 40).sum(axis=1) >= 6).astype(int)
    features, target = dataframe.drop(columns=["pass_fail"]), dataframe["pass_fail"]
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(features, target)
    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(features.columns.tolist(), os.path.join(BASE_DIR, "model_columns.pkl"))
    return model


def ensure_model_artifacts():
    if not all(os.path.exists(os.path.join(BASE_DIR, artifact)) for artifact in ARTIFACTS):
        train_model()


if __name__ == "__main__":
    train_model()
    print("EduPredict model and encoders trained successfully.")
