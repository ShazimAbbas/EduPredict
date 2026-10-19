import io
import os
from datetime import datetime, timezone
from functools import wraps

import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash

from model_engine import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SCORE_COLUMNS, predict_student_performance
from train_model import ensure_model_artifacts
from database import get_collections

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-before-production")
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ensure_model_artifacts()

users, students, predictions = get_collections()
UPLOAD_REQUIRED = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


def seed_database():
    users.create_index([("email", ASCENDING)], unique=True)
    students.create_index([("email", ASCENDING)], unique=True)
    students.create_index([("student_id", ASCENDING)], unique=True)
    for name, email, role in [
        ("System Administrator", "admin@edupredict.com", "admin"),
        ("Lead Teacher", "teacher@edupredict.com", "teacher"),
    ]:
        users.update_one({"email": email}, {"$setOnInsert": {
            "username": name, "email": email, "password": generate_password_hash("EduPredict123!"),
            "role": role, "created_at": datetime.now(timezone.utc),
        }}, upsert=True)
    users.delete_many({"role": {"$nin": ["admin", "teacher", "student"]}})


def login_required(*roles):
    def decorate(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_email" not in session:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("signin"))
            if roles and session.get("role") not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorate


def new_student_profile(name, email, teacher="teacher@edupredict.com"):
    identity = datetime.now().strftime("STU-%y%m%d%H%M%S%f")
    return {"student_id": identity, "name": name, "email": email, "assigned_teacher": teacher,
            "gender": "unknown", "part_time_job": False, "extracurricular_activities": False,
            "career_aspiration": "unknown", "attendance_percentage": 0, "lms_logins_last_30_days": 0,
            "previous_gpa": 0, **{column: 0 for column in NUMERIC_COLUMNS}}


def save_prediction(student, generated_by):
    result = predict_student_performance(student)
    predictions.insert_one({"student_id": student["student_id"], "student_email": student["email"],
                            "student_name": student["name"], "generated_by": generated_by,
                            "timestamp": datetime.now(timezone.utc), **result})


def latest_by_student(roster):
    return {student["email"]: predictions.find_one({"student_email": student["email"]}, sort=[("timestamp", -1)]) for student in roster}


def build_upload_record(row, index):
    name = str(row.get("student_name") or row.get("name") or f"Student {index + 1}").strip()
    email = str(row.get("email") or f"upload.{datetime.now().strftime('%Y%m%d%H%M%S')}.{index}@edupredict.local").lower()
    record = new_student_profile(name, email, session["user_email"])
    record["student_id"] = str(row.get("student_id") or f"UPL-{datetime.now().strftime('%y%m%d%H%M%S')}-{index + 1}")
    for column in CATEGORICAL_COLUMNS:
        record[column] = str(row[column]).strip()
    for column in NUMERIC_COLUMNS:
        record[column] = float(row[column])
    return record


@app.context_processor
def global_context():
    return {"current_user": {"name": session.get("username"), "role": session.get("role"), "email": session.get("user_email")}}


@app.route("/")
def home():
    return redirect(url_for("dashboard")) if session.get("user_email") else redirect(url_for("signin"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name, email, password = request.form.get("name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter your name, email, and a password of at least 8 characters.", "danger")
        else:
            try:
                users.insert_one({"username": name, "email": email, "password": generate_password_hash(password), "role": "student", "created_at": datetime.now(timezone.utc)})
                students.update_one({"email": email}, {"$setOnInsert": new_student_profile(name, email)}, upsert=True)
                flash("Your student account is ready. Please sign in.", "success")
                return redirect(url_for("signin"))
            except DuplicateKeyError:
                flash("An account with that email already exists.", "danger")
    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = users.find_one({"email": email})
        if user and check_password_hash(user["password"], request.form.get("password", "")):
            session.clear(); session.update(user_email=user["email"], username=user["username"], role=user["role"])
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("signin.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Compatibility route for the reference project's login URL."""
    return signin()


@app.route("/logout")
def logout():
    session.clear(); flash("You have been signed out.", "success")
    return redirect(url_for("signin"))


@app.route("/dashboard")
@login_required("admin", "teacher", "student")
def dashboard():
    if session["role"] == "admin":
        report = list(predictions.aggregate([{"$group": {"_id": "$generated_by", "total": {"$sum": 1}, "pass_count": {"$sum": {"$cond": [{"$eq": ["$pass_fail", "Pass"]}, 1, 0]}}}}, {"$sort": {"total": -1}}]))
        subject_reports = []
        for column in SCORE_COLUMNS:
            values = [student.get(column) for student in students.find({column: {"$exists": True}}, {column: 1}) if isinstance(student.get(column), (int, float))]
            if values: subject_reports.append({"subject": column.replace("_score", "").title(), "average": round(sum(values) / len(values), 1), "count": len(values)})
        return render_template("admin_dashboard.html", user_count=users.count_documents({}), student_count=students.count_documents({}), prediction_count=predictions.count_documents({}), reports=report, subject_reports=subject_reports, recent_predictions=list(predictions.find().sort("timestamp", -1).limit(12)))
    if session["role"] == "teacher":
        roster = list(students.find({"assigned_teacher": session["user_email"]}).sort("name", 1))
        return render_template("teacher_dashboard.html", roster=roster, latest_predictions=latest_by_student(roster))
    student = students.find_one({"email": session["user_email"]})
    return render_template("student_dashboard.html", student=student, student_predictions=list(predictions.find({"student_email": session["user_email"]}).sort("timestamp", -1)))


@app.route("/teacher/predict/<student_id>", methods=["POST"])
@login_required("teacher")
def teacher_predict(student_id):
    student = students.find_one({"student_id": student_id, "assigned_teacher": session["user_email"]})
    if not student: abort(403)
    save_prediction(student, session["user_email"])
    flash(f"Prediction generated for {student['name']}.", "success")
    return redirect(url_for("dashboard"))


@app.route("/teacher/upload_csv", methods=["POST"])
@login_required("teacher")
def upload_csv():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        flash("Upload a CSV file using the required student schema.", "danger")
        return redirect(url_for("dashboard"))
    try:
        dataframe = pd.read_csv(io.TextIOWrapper(uploaded.stream, encoding="utf-8-sig"))
        missing = [column for column in UPLOAD_REQUIRED if column not in dataframe.columns]
        if missing: raise ValueError("Missing columns: " + ", ".join(missing))
        dataframe = dataframe.dropna(subset=UPLOAD_REQUIRED)
        processed = 0
        for index, row in dataframe.iterrows():
            record = build_upload_record(row, index)
            students.update_one({"email": record["email"]}, {"$set": record}, upsert=True)
            save_prediction(record, session["user_email"])
            processed += 1
        flash(f"Imported and predicted {processed} student record(s).", "success")
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
        flash(f"CSV could not be processed: {error}", "danger")
    return redirect(url_for("dashboard"))


@app.route("/teacher/update_student/<student_id>", methods=["GET", "POST"])
@login_required("teacher")
def update_student(student_id):
    student = students.find_one({"student_id": student_id, "assigned_teacher": session["user_email"]})
    if not student: abort(403)
    if request.method == "POST":
        try:
            attendance, logins, gpa = float(request.form["attendance_percentage"]), float(request.form["lms_logins_last_30_days"]), float(request.form["previous_gpa"])
            if not 0 <= attendance <= 100 or logins < 0 or not 0 <= gpa <= 4: raise ValueError
            students.update_one({"_id": student["_id"]}, {"$set": {"attendance_percentage": attendance, "lms_logins_last_30_days": logins, "previous_gpa": gpa}})
            flash("Student metrics updated successfully.", "success"); return redirect(url_for("dashboard"))
        except (KeyError, ValueError): flash("Use valid metric values.", "danger")
    return render_template("update_student.html", student=student)


@app.route("/admin/users")
@login_required("admin")
def manage_users():
    return render_template("manage_users.html", all_users=list(users.find().sort("created_at", -1)))


@app.route("/admin/add_user", methods=["POST"])
@login_required("admin")
def add_user():
    name, email, password, role = request.form.get("name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("password", ""), request.form.get("role", "")
    if not name or not email or len(password) < 8 or role not in {"admin", "teacher", "student"}:
        flash("Enter valid account details and an allowed role.", "danger")
    else:
        try:
            users.insert_one({"username": name, "email": email, "password": generate_password_hash(password), "role": role, "created_at": datetime.now(timezone.utc)})
            if role == "student": students.update_one({"email": email}, {"$setOnInsert": new_student_profile(name, email)}, upsert=True)
            flash("User added.", "success")
        except DuplicateKeyError: flash("An account with that email already exists.", "danger")
    return redirect(url_for("manage_users"))


@app.route("/admin/users/<user_id>/role", methods=["POST"])
@login_required("admin")
def update_role(user_id):
    from bson import ObjectId
    role = request.form.get("role")
    if role not in {"admin", "teacher", "student"}: abort(400)
    users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})
    flash("User role updated.", "success"); return redirect(url_for("manage_users"))


@app.route("/admin/users/<user_id>/delete", methods=["POST"])
@login_required("admin")
def delete_user(user_id):
    from bson import ObjectId
    target = users.find_one({"_id": ObjectId(user_id)})
    if target and target["email"] != session["user_email"]:
        users.delete_one({"_id": target["_id"]}); flash("User removed.", "success")
    else: flash("You cannot delete your own account.", "danger")
    return redirect(url_for("manage_users"))


@app.errorhandler(403)
def forbidden(_): return render_template("error.html", code=403, message="You do not have permission to view that page."), 403


seed_database()

if __name__ == "__main__": app.run(debug=True)
