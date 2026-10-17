import os
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash
from model_engine import predict_student_performance

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-before-production")
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(app.config["MONGO_URI"], serverSelectionTimeoutMS=3000)
db = client["edupredict"]
users = db.users
students = db.students
predictions = db.predictions
support = db.support


def seed_database():
    """Create indexes and the staff accounts required by the application."""
    users.create_index([("email", ASCENDING)], unique=True)
    students.create_index([("email", ASCENDING)], unique=True)
    students.create_index([("student_id", ASCENDING)], unique=True)
    for name, email, role in [
        ("System Administrator", "admin@edupredict.com", "admin"),
        ("Data Analyst", "analyst@edupredict.com", "analyst"),
        ("Lead Teacher", "teacher@edupredict.com", "teacher"),
    ]:
        if not users.find_one({"email": email}):
            users.insert_one({
                "username": name,
                "email": email,
                "password": generate_password_hash("EduPredict123!"),
                "role": role,
                "created_at": datetime.now(timezone.utc),
            })


def login_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_email" not in session:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("signin"))
            if roles and session.get("role") not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


@app.context_processor
def global_context():
    return {"current_user": {"name": session.get("username"), "role": session.get("role"), "email": session.get("user_email")}}


@app.route("/")
def home():
    return redirect(url_for("dashboard")) if session.get("user_email") else redirect(url_for("signin"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter your name, a valid email, and a password of at least 8 characters.", "danger")
        else:
            try:
                users.insert_one({"username": name, "email": email, "password": generate_password_hash(password), "role": "student", "created_at": datetime.now(timezone.utc)})
                students.update_one({"email": email}, {"$setOnInsert": {"student_id": f"STU-{datetime.now().strftime('%y%m%d%H%M%S%f')}", "name": name, "email": email, "attendance_percentage": 0.0, "lms_logins_last_30_days": 0.0, "previous_gpa": 0.0, "midterm_score": 0.0, "assigned_teacher": "teacher@edupredict.com"}}, upsert=True)
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
            session.clear()
            session.update(user_email=user["email"], username=user["username"], role=user["role"])
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("signin"))


@app.route("/dashboard")
@login_required("admin", "analyst", "teacher", "student")
def dashboard():
    role = session["role"]
    if role == "admin":
        return render_template("admin_dashboard.html", user_count=users.count_documents({}), student_count=students.count_documents({}), prediction_count=predictions.count_documents({}), recent_predictions=list(predictions.find().sort("timestamp", -1).limit(10)))
    if role == "analyst":
        return render_template("analyst_dashboard.html", student_count=students.count_documents({}), recent_predictions=list(predictions.find().sort("timestamp", -1).limit(8)))
    if role == "teacher":
        roster = list(students.find({"assigned_teacher": session["user_email"]}).sort("name", 1))
        return render_template("teacher_dashboard.html", roster=roster, latest_predictions=latest_by_student(roster))
    student = students.find_one({"email": session["user_email"]})
    student_predictions = list(predictions.find({"student_email": session["user_email"]}).sort("timestamp", -1))
    return render_template("student_dashboard.html", student=student, student_predictions=student_predictions)


def latest_by_student(roster):
    result = {}
    for student in roster:
        result[student["email"]] = predictions.find_one({"student_email": student["email"]}, sort=[("timestamp", -1)])
    return result


@app.route("/teacher/update_student/<student_id>", methods=["GET", "POST"])
@login_required("teacher")
def update_student(student_id):
    student = students.find_one({"student_id": student_id})
    if not student:
        abort(404)
    if student.get("assigned_teacher") not in (None, session["user_email"]):
        abort(403)
    if request.method == "POST":
        try:
            attendance = float(request.form["attendance_percentage"])
            logins = float(request.form["lms_logins_last_30_days"])
            gpa = float(request.form["previous_gpa"])
        except (KeyError, ValueError):
            flash("Metrics must be valid numeric values.", "danger")
        else:
            if not 0 <= attendance <= 100 or logins < 0 or not 0 <= gpa <= 4:
                flash("Attendance must be 0–100, logins cannot be negative, and GPA must be 0–4.", "danger")
            else:
                students.update_one({"_id": student["_id"]}, {"$set": {"attendance_percentage": attendance, "lms_logins_last_30_days": logins, "previous_gpa": gpa}})
                flash("Student metrics updated successfully.", "success")
                return redirect(url_for("dashboard"))
    return render_template("update_student.html", student=student)


@app.route("/analyst/predict", methods=["GET", "POST"])
@login_required("analyst", "teacher")
def analyst_predict():
    # Analysts can process the whole dataset; teachers can process only their class.
    query = {} if session["role"] == "analyst" else {"assigned_teacher": session["user_email"]}
    roster = list(students.find(query).sort("name", 1))
    if request.method == "POST":
        selected = request.form.getlist("student_email") or [item["email"] for item in roster]
        created = 0
        prediction_query = {"email": {"$in": selected}}
        if session["role"] == "teacher":
            prediction_query["assigned_teacher"] = session["user_email"]
        for student in students.find(prediction_query):
            result = predict_student_performance(student)
            predictions.insert_one({"student_id": student["student_id"], "student_email": student["email"], "predicted_gpa": result["predicted_gpa"], "predicted_performance": result["predicted_performance"], "dropout_risk": result["dropout_risk"], "generated_by": session["user_email"], "timestamp": datetime.now(timezone.utc)})
            created += 1
        flash(f"Generated and broadcast {created} prediction result(s).", "success")
        return redirect(url_for("analyst_predict"))
    return render_template("prediction_portal.html", roster=roster, recent_predictions=list(predictions.find().sort("timestamp", -1).limit(20)))


@app.route("/admin/users")
@login_required("admin")
def manage_users():
    return render_template("manage_users.html", all_users=list(users.find().sort("created_at", -1)))


@app.route("/admin/users/<user_id>/role", methods=["POST"])
@login_required("admin")
def update_role(user_id):
    role = request.form.get("role")
    if role not in {"admin", "analyst", "teacher", "student"}:
        abort(400)
    from bson import ObjectId
    users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})
    flash("User role updated.", "success")
    return redirect(url_for("manage_users"))


@app.route("/admin/users/<user_id>/delete", methods=["POST"])
@login_required("admin")
def delete_user(user_id):
    from bson import ObjectId
    target = users.find_one({"_id": ObjectId(user_id)})
    if target and target["email"] != session["user_email"]:
        users.delete_one({"_id": target["_id"]})
        flash("User removed.", "success")
    else:
        flash("You cannot delete your own account.", "danger")
    return redirect(url_for("manage_users"))


@app.route("/support", methods=["GET", "POST"])
@login_required("admin", "analyst", "teacher", "student")
def support_ticket():
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            support.insert_one({"user_email": session["user_email"], "message": message, "status": "open", "timestamp": datetime.now(timezone.utc)})
            flash("Your support request has been sent.", "success")
            return redirect(url_for("support_ticket"))
    tickets = list(support.find({"user_email": session["user_email"]}).sort("timestamp", -1)) if session["role"] != "admin" else list(support.find().sort("timestamp", -1).limit(30))
    return render_template("support.html", tickets=tickets)


@app.errorhandler(403)
def forbidden(_):
    return render_template("error.html", code=403, message="You do not have permission to view that page."), 403


# Execute this during application startup as well as when launched by a WSGI server.
# Both operations are idempotent, so the Flask development reloader is safe.
seed_database()

if __name__ == "__main__":
    app.run(debug=True)
