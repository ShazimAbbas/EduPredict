"""Seed EduPredict with admin and teacher accounts plus CSV students."""
import os
from datetime import datetime, timezone

import pandas as pd
from pymongo import ASCENDING, MongoClient
from werkzeug.security import generate_password_hash

from model_engine import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "student_scores_dataset.csv")


def seed(limit=50):
    db = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/")).edupredict
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.students.create_index([("email", ASCENDING)], unique=True)
    db.students.create_index([("student_id", ASCENDING)], unique=True)
    for name, email, role in [("System Administrator", "admin@edupredict.com", "admin"), ("Lead Teacher", "teacher@edupredict.com", "teacher")]:
        db.users.update_one({"email": email}, {"$setOnInsert": {"username": name, "email": email, "password": generate_password_hash("EduPredict123!"), "role": role, "created_at": datetime.now(timezone.utc)}}, upsert=True)
    db.users.delete_many({"role": {"$nin": ["admin", "teacher", "student"]}})
    if not os.path.exists(DATASET_PATH):
        print("Default accounts created. student_scores_dataset.csv was not found.")
        return
    for _, row in pd.read_csv(DATASET_PATH).head(limit).iterrows():
        name = f"{row['first_name']} {row['last_name']}"
        record = {"student_id": str(row["id"]), "name": name, "email": str(row["email"]).lower(), "assigned_teacher": "teacher@edupredict.com", **{column: row[column] for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS}}
        db.students.update_one({"email": record["email"]}, {"$set": record}, upsert=True)
    print(f"Seeded {min(limit, len(pd.read_csv(DATASET_PATH)))} students.")


if __name__ == "__main__":
    seed()
