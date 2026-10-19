# EduPredict

EduPredict is a Flask, MongoDB, and scikit-learn education analytics application for Admins, Teachers, and Students.

## Run locally

1. Start MongoDB at `mongodb://localhost:27017/` or configure `MONGO_URI`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python seed_db.py` to import sample students from `student_scores_dataset.csv`.
4. Start the application with `python flask_app.py`.

Default accounts use the password `EduPredict123!`:

| Email | Role |
| --- | --- |
| admin@edupredict.com | Admin |
| teacher@edupredict.com | Teacher |

## Teacher batch upload

Teachers can upload a CSV from their dashboard. Use [sample_batch.csv](D:\Projects\EduPredict\sample_batch.csv) as the header reference. Each upload updates student records, runs the pass/fail model, and saves expected marks, risk, and improvement suggestions.

## Model artifacts

The application automatically trains missing model artifacts from `student_scores_dataset.csv` during startup. To force a retrain after replacing the dataset, run `python train_model.py` and then restart `python flask_app.py`.
