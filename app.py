import csv
import io
import os
from datetime import datetime, timezone
from functools import wraps

import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for
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
    for name, email, role, password in [
        ("System Administrator", "admin@edupredict.com", "admin", "admin123"),
        ("Lead Teacher", "teacher@edupredict.com", "teacher", "teacher123"),
    ]:
        users.update_one({"email": email}, {"$set": {
            "username": name, "email": email, "password": generate_password_hash(password),
            "role": role, "updated_at": datetime.now(timezone.utc),
        }, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}}, upsert=True)
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


def teacher_prediction_count():
    return predictions.count_documents({"generated_by": session["user_email"]})


def build_upload_record(row, index):
    name = str(row.get("student_name") or row.get("student") or row.get("name") or f"Student {index + 1}").strip()
    email = str(row.get("email") or f"upload.{datetime.now().strftime('%Y%m%d%H%M%S')}.{index}@edupredict.local").lower()
    record = new_student_profile(name, email, session["user_email"])
    record["student_id"] = str(row.get("student_id") or f"UPL-{datetime.now().strftime('%y%m%d%H%M%S')}-{index + 1}")
    for column in CATEGORICAL_COLUMNS:
        record[column] = str(row[column]).strip()
    for column in NUMERIC_COLUMNS:
        record[column] = float(row[column])
    # Optional columns in the updated teacher template enrich the student profile.
    for column in ("previous_gpa", "lms_logins_last_30_days", "attendance_percentage"):
        if column in row and pd.notna(row[column]):
            record[column] = float(row[column])
    record["attendance_percentage"] = int(record["attendance_percentage"])
    record["lms_logins_last_30_days"] = int(record["lms_logins_last_30_days"])
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
        pass_count = predictions.count_documents({"pass_fail": "Pass"})
        fail_count = predictions.count_documents({"pass_fail": "Fail"})
        total_results = pass_count + fail_count
        pass_rate = round(pass_count / total_results * 100) if total_results else 0
        subject_reports = []
        for column in SCORE_COLUMNS:
            values = [student.get(column) for student in students.find({column: {"$exists": True}}, {column: 1}) if isinstance(student.get(column), (int, float))]
            if values: subject_reports.append({"subject": column.replace("_score", "").title(), "average": round(sum(values) / len(values), 1), "count": len(values)})
        return render_template("admin_dashboard.html", user_count=users.count_documents({}), student_count=students.count_documents({}), prediction_count=predictions.count_documents({}), reports=report, subject_reports=subject_reports, recent_predictions=list(predictions.find().sort("timestamp", -1)), pass_count=pass_count, fail_count=fail_count, pass_rate=pass_rate)
    if session["role"] == "teacher":
        roster = list(students.find({"assigned_teacher": session["user_email"]}).sort("name", 1))
        return render_template("teacher_dashboard.html", roster=roster, latest_predictions=latest_by_student(roster), prediction_count=teacher_prediction_count())
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


@app.route("/teacher/predict_selected", methods=["POST"])
@login_required("teacher")
def teacher_predict_selected():
    selected_ids = request.form.getlist("student_ids")
    if not selected_ids:
        flash("Select at least one student to generate predictions.", "warning")
        return redirect(url_for("dashboard"))
    roster = students.find({"student_id": {"$in": selected_ids}, "assigned_teacher": session["user_email"]})
    processed = 0
    for student in roster:
        save_prediction(student, session["user_email"])
        processed += 1
    flash(f"Generated predictions for {processed} student(s).", "success")
    return redirect(url_for("dashboard"))


@app.route("/teacher/download_predictions")
@login_required("teacher")
def download_predictions():
    prediction_rows = list(predictions.find({"generated_by": session["user_email"]}).sort("timestamp", -1))
    if not prediction_rows:
        flash("Generate a prediction before downloading a report.", "warning")
        return redirect(url_for("dashboard"))
    output = io.StringIO()
    columns = ["student_id", "name", "email", *CATEGORICAL_COLUMNS, "absence_days", "weekly_self_study_hours", *SCORE_COLUMNS,
               "previous_gpa", "lms_logins_last_30_days", "attendance_percentage", "expected_marks", "pass_fail", "dropout_risk", "confidence", "improvement_suggestion", "generated_by", "timestamp"]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for prediction in prediction_rows:
        student = students.find_one({"student_id": prediction["student_id"]}) or {}
        writer.writerow({**student, **prediction})
    payload = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    return send_file(payload, mimetype="text/csv", as_attachment=True, download_name="edupredict_predictions.csv")


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
            values = {"absence_days": int(request.form["absence_days"]), "weekly_self_study_hours": int(request.form["weekly_self_study_hours"]),
                      **{column: int(request.form[column]) for column in SCORE_COLUMNS}, "previous_gpa": float(request.form["previous_gpa"]),
                      "lms_logins_last_30_days": int(request.form["lms_logins_last_30_days"]), "attendance_percentage": int(request.form["attendance_percentage"])}
            if values["absence_days"] < 0 or values["weekly_self_study_hours"] < 0 or values["lms_logins_last_30_days"] < 0 or not 0 <= values["previous_gpa"] <= 4 or not 0 <= values["attendance_percentage"] <= 100 or any(not 0 <= values[column] <= 100 for column in SCORE_COLUMNS): raise ValueError
            students.update_one({"_id": student["_id"]}, {"$set": values})
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


@app.route("/admin/reset_predictions", methods=["POST"])
@login_required("admin")
def reset_predictions():
    result = predictions.delete_many({})
    flash(f"Cleared {result.deleted_count} generated prediction(s). Student records were kept.", "success")
    return redirect(url_for("dashboard"))


@app.route("/teacher/reset_predictions", methods=["POST"])
@login_required("teacher")
def teacher_reset_predictions():
    result = predictions.delete_many({"generated_by": session["user_email"]})
    flash(f"Cleared {result.deleted_count} prediction(s) generated by you. Student records were kept.", "success")
    return redirect(url_for("dashboard"))


@app.route("/admin/remove_students", methods=["POST"])
@login_required("admin")
def remove_students():
    result = students.delete_many({})
    flash(f"Removed {result.deleted_count} student record(s). User accounts and predictions were kept.", "success")
    return redirect(url_for("dashboard"))


@app.errorhandler(403)
def forbidden(_): return render_template("error.html", code=403, message="You do not have permission to view that page."), 403


seed_database()

if __name__ == "__main__": app.run(debug=True)
