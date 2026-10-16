"""Seed EduPredict's MongoDB collections from the local EduPredict.csv file."""
import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from pymongo import ASCENDING, MongoClient
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "EduPredict.csv")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")


def first_present(row, names, default=""):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def seed(clear=False, limit=25):
    client = MongoClient(MONGO_URI)
    db = client.edupredict
    if clear:
        db.users.delete_many({})
        db.students.delete_many({})
        db.predictions.delete_many({})
        db.support.delete_many({})
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.students.create_index([("email", ASCENDING)], unique=True)
    db.students.create_index([("student_id", ASCENDING)], unique=True)
    for username, email, role in [
        ("System Administrator", "admin@edupredict.com", "admin"),
        ("Data Analyst", "analyst@edupredict.com", "analyst"),
        ("Lead Teacher", "teacher@edupredict.com", "teacher"),
    ]:
        db.users.update_one({"email": email}, {"$setOnInsert": {"username": username, "email": email, "password": generate_password_hash("EduPredict123!"), "role": role, "created_at": datetime.now(timezone.utc)}}, upsert=True)
    if not os.path.exists(CSV_PATH):
        print("Default users created. EduPredict.csv was not found; no student records were seeded.")
        return
    dataframe = pd.read_csv(CSV_PATH).head(limit)
    for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
        student_id = str(first_present(row, ["student_id", "id"], f"CSV-{position:04d}"))
        name = str(first_present(row, ["name", "student_name"], f"Student {position}"))
        email = str(first_present(row, ["email", "student_email"], f"student{position}@edupredict.local")).lower()
        attendance = float(first_present(row, ["attendance_percentage", "attendance_rate"], 0))
        logins = float(first_present(row, ["lms_logins_last_30_days", "lms_activity_score"], 0))
        gpa = float(first_present(row, ["previous_gpa", "gpa"], 0))
        db.students.update_one({"student_id": student_id}, {"$set": {"student_id": student_id, "name": name, "email": email, "attendance_percentage": attendance, "lms_logins_last_30_days": logins, "previous_gpa": gpa, "midterm_score": float(first_present(row, ["midterm_score"], 0)), "assigned_teacher": "teacher@edupredict.com"}}, upsert=True)
    print(f"Seeded default users and {len(dataframe)} student record(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear EduPredict collections before seeding.")
    parser.add_argument("--limit", type=int, default=25, help="Number of CSV rows to import.")
    arguments = parser.parse_args()
    seed(clear=arguments.clear, limit=arguments.limit)
