import os
import re
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier   # pip install xgboost
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Only needed if using SentenceTransformer embeddings
from sentence_transformers import SentenceTransformer

import joblib
from tqdm import tqdm
tqdm.pandas()

#-------- clean text material for cleaning the articles text...
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

import nltk
# nltk.download('stopwords')

nltk_stopwords = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in nltk_stopwords]
    return ' '.join(tokens)

#---------------------------------------------------------------

models_dir="./fakenews_datasets/models"

# for the moment training is done only with Kaggle Fakenews dataset ~ 45000 articles, possible extention to FakeNewsNet ~ 25000 articles !!!!
# def Create_Models_from_dataset(models_dir="./fakenews_datasets/models", dataset_path="", use_embeddings=False):

# def Create_Models_from_mongodb(models_dir=models_dir, use_embeddings=False):
# changed to process only the selected articles...
def Train_Models_from_dataframe(df, filtered_option=True, models_dir=models_dir, use_embeddings=False):
    # filtered_option=True if Train_models.. on filtered articles...
    # open selected dataset
    # df = pd.read_csv(dataset_path)
    import pandas as pd
    from pymongo import MongoClient
    import swifter  # improves the speed of clean_text

    def Save_model_metadata(model_name, use_embeddings=False):
        # from pymongo import MongoClient
        from datetime import datetime

        use_tfidf=not use_embeddings
        client = MongoClient("mongodb://localhost:27017/")
        db = client["realtime_news_db"]
        models_col = db["models"]
        # insert a new record for the trained model
        models_col.insert_one({
            "model_name": model_name,
            "timestamp": datetime.now(),
            "use_embeddings": use_embeddings,
            "use_tfidf": use_tfidf
        })
        print(f"Saving model metadata for {model_name}...")

    # if analyse all articles collection from MongoDB...
    if not filtered_option:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["realtime_news_db"]
        articles_col = db["articles"]
        # Only selects (filters) labeled data (0/1) and ignore those without text...
        print("importing  MongoDB collection of articles..")
        articles = list(articles_col.find({"text": {"$ne": None}, "label": {"$in": [0, 1]}}))
        print("articles imported from MongoDB collection.")
        '''
        this script can replace the created list : articles = list(articles_col.find({"text": {"$ne": None}, "label": {"$in": [0, 1]}}))
        if the number of articles > 100.000 where there may be memory overflow...
        cursor = articles_col.find(
            {"text": {"$ne": None}, "label": {"$in": [0, 1]}},
            projection={"_id": 1, "title" : 1, "text": 1, "label": 1}
        ).batch_size(500)
        # Process each article here one by one without storing all in memory
        for article in cursor: process_article(article)  # <-- define your processing logic here
        '''
        # Convert to DataFrame
        df = pd.DataFrame(articles)  # this is deleted because we want to run analysis on filtered articles...

    if df.empty:
        print("❌ No articles provided for model training.")
        return

    df = df.loc[df["text"].str.len() > 8].copy()
    y = pd.to_numeric(df["label"], errors="coerce")
    if set(y.dropna().unique()) != {0, 1} or y.isna().any() or y.value_counts().min() < 5:
        raise ValueError("Training requires at least five usable articles for each label (0 and 1).")
    y = y.astype(int)
    X = df["text"].swifter.apply(clean_text)

    models = [
        ("log_model", "Logistic Regression", LogisticRegression(max_iter=1000)),
        ("rf_model", "Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
    suffix = ""
    if use_embeddings:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        X = embedder.encode(X.tolist(), show_progress_bar=True)
        suffix = "_embed"
    else:
        models.append(("xgb_model", "XGBoost", XGBClassifier(eval_metric="logloss", verbosity=0)))
        models = [
            (key, name, Pipeline([("tfidf", TfidfVectorizer(max_features=5000)), ("clf", model)]))
            for key, name, model in models
        ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(models_dir, exist_ok=True)
    metrics = []
    curves = []

    for key, name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        print(f"\n{name}")
        print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions, zero_division=0))
        print(f"Holdout ROC AUC: {roc_auc:.3f}")
        print(f"Cross-validation F1: {cv_scores} (mean {cv_scores.mean():.3f})")
        metrics.append((
            precision_score(y_test, predictions, zero_division=0),
            recall_score(y_test, predictions, zero_division=0),
            f1_score(y_test, predictions, zero_division=0),
        ))
        curves.append((fpr, tpr, f"{name} (AUC = {roc_auc:.3f})"))
        path = os.path.join(models_dir, f"{key}{suffix}.pkl")
        joblib.dump(model, path)
        Save_model_metadata(path, use_embeddings)

    positions = np.arange(len(models))
    width = 0.2
    plt.figure(figsize=(8, 5))
    for offset, label, scores in zip((-width, 0, width), ("Precision", "Recall", "F1"), zip(*metrics)):
        plt.bar(positions + offset, scores, width, label=label)
    plt.ylabel("Score")
    plt.title("Model Metrics Comparison")
    plt.xticks(positions, [name for _, name, _ in models])
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    for fpr, tpr, label in curves:
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
