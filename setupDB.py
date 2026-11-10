# this script only exist individually so it can reset the database and add one article, a few users and subjects.
#
import argparse
from pymongo import MongoClient    # pip install
from datetime import datetime
import uuid

# --- MongoDB Connection ---
client = MongoClient("mongodb://localhost:27017/")
db = client["realtime_news_db"]

articles_col = db["articles"]
users_col = db["users"]
subjects_col = db["subjects"]
models_col = db["models"]

def setup_sample_data():
    db.articles.drop()
    db.users.drop()
    db.comments.drop()
    db.subjects.drop()
    db.models.drop()
    db.user_logs.drop()

    user_logs = {"user_email": "unknown", "timestamp": datetime.now(), "action": "unknown", "description": "unknown"}
    db.user_logs.insert_one(user_logs)

    subjects = ["Politics", "Health", "Technology", "Economy", "Environment", "Culture"]
    db.subjects.insert_many([{"name": s} for s in subjects])

    models = {"model_name": "fake (init) model 1", "timestamp": datetime.now(), "use_embeddings": True, "use_tfidf": False}
    db.models.insert_one(models)

    users = [
        {"_id": uuid.uuid4().hex, "name": "Alice Moderator", "email": "alice@newsapp.com",      "role": "moderator",   "password": "alice123",  "joined": datetime.now()},
        {"_id": uuid.uuid4().hex, "name": "Bob Admin",       "email": "admin@example.test",        "role": "admin",       "password": "local-demo",       "joined": datetime.now()},
        {"_id": uuid.uuid4().hex, "name": "Charlie User",    "email": "charlie@newsapp.com",    "role": "user",        "password": "charlie",   "joined": datetime.now()},
    ]
    db.users.insert_many(users)

    db.articles.insert_one({
        "title": "Aliens land in New York!",
        "text": "Today, unidentified spacecraft appeared in the skies of NYC...",
        "subject": "Culture",
        "source": "Twitter",
        "date": datetime.now(),
        "label": 0,
        #"confidence": 0.12,
        "polarity": 0,
        "subjectivity": 0,
        "v_compound": 0,
        "sentiment": "neutral",
        #"score": -0.3,
        "submitted_by": users[2]["_id"],
        "reviewed": False,
        "comments": []
    })

    print("Demo data initialized. Sign in as admin@example.test with password local-demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a local MongoDB database for the demo.")
    parser.add_argument("--reset", action="store_true", help="Replace collections in an existing demo database.")
    args = parser.parse_args()

    if db.list_collection_names() and not args.reset:
        parser.error("The demo database already has collections. Use --reset to replace its contents.")
    setup_sample_data()
