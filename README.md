# EduPredict

EduPredict is a Flask and MongoDB educational analytics platform with role-based dashboards and a scikit-learn GPA/dropout-risk forecasting workflow.

## Quick start

1. Ensure MongoDB is running locally at `mongodb://localhost:27017/` (or set `MONGO_URI`).
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Visit `http://127.0.0.1:5000`.

Staff accounts are seeded at first startup. The shared initial password is `EduPredict123!`:

| Account | Role |
| --- | --- |
| admin@edupredict.com | Admin |
| analyst@edupredict.com | Analyst |
| teacher@edupredict.com | Teacher |

Public sign-up deliberately has no role field and always creates a student account. Set `FLASK_SECRET_KEY` to a unique secret before deployment.

## CSV data and model setup

Put `EduPredict.csv` in the project root, then run `python seed_db.py` to add the first 25 student records. Use `python seed_db.py --clear` only when you intentionally want to reset the application collections. Open [EduPredict_Model_Training.ipynb](D:\Projects\EduPredict\EduPredict_Model_Training.ipynb) in Jupyter or Colab and run its cells in order to train and save `edupredict_model.pkl` and `scaler.pkl`. The application uses those saved artifacts for predictions and retains a simple fallback score until they are available.

## Data flow

The analyst selects student records in the Prediction Portal. The model creates a GPA forecast and a Low/Medium/High dropout assessment, then writes each result to `predictions`. The teacher, student, and admin dashboards query that same collection live on every request.
