"""MongoDB connection and collection access for EduPredict."""
import os

from pymongo import MongoClient


def get_collections():
    client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=3000)
    database = client.edupredict
    return database.users, database.students, database.predictions
