# run 2 types of sentiment analysis on a set of articles
# 'df=None,'  must be the 1st parameter of sentiment_analysis... if must run in filtered
def sentiment_analysis_on_articles(df=None, save_in_mongoDB=True,
                                   save_csv_path="./fakenews_datasets/sentiment/articles_sentiments.csv"):

    import pandas as pd
    from textblob import TextBlob
    import seaborn as sns
    import nltk
    import matplotlib.pyplot as plt
    from nltk.sentiment import SentimentIntensityAnalyzer
    from tqdm import tqdm
    import os
    import re
    from bs4 import BeautifulSoup
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy.stats import ks_2samp
    from pymongo import MongoClient

    nltk.download("vader_lexicon")
    tqdm.pandas()

    # special clean_text for sentiment analysis
    def clean_text_for_sentiment(text):
        if pd.isna(text):
            return ""
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    print("📦 Run Sentiment Analysis...")
    print("📦 Loading articles from Mongodb collection...")
    # df = pd.read_csv(DataFileName)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["realtime_news_db"]
    articles_col = db["articles"]

    # Load from MongoDB
    if df is None:
        cursor = articles_col.find({}, {"_id": 1, "title": 1, "text": 1, "subject": 1, "label": 1})
        df = pd.DataFrame(list(cursor))
    if df.empty:
        print("❌ No articles provided for sentiment analysis.")
        return

    print("🧼 Cleaning and combining text...")
    df["clean_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    df["clean_text"] = df["clean_text"].progress_apply(clean_text_for_sentiment)  # added recently for additional text cleaning

    print("🧠 Running TextBlob analysis...")
    df[["tb_polarity", "tb_subjectivity"]] = df["clean_text"].progress_apply(
        lambda x: pd.Series(TextBlob(x).sentiment)
    )

    print("🧠 Running VADER analysis...")
    sia = SentimentIntensityAnalyzer()
    df[["vader_neg", "vader_neu", "vader_pos", "vader_compound"]] = df["clean_text"].progress_apply(
        lambda x: pd.Series(sia.polarity_scores(x))
    )

    # --- Visualization 1: TextBlob polarity distribution by label ---
    print("📊 TextBlob: Polarity distribution by label...")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="tb_polarity", hue="label", bins=30, kde=True)
    plt.xlabel("TextBlob Polarity")
    plt.title("TextBlob Polarity Distribution by Label")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: VADER compound score distribution by label ---
    print("📊 VADER: Compound distribution by label...")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="vader_compound", hue="label", bins=30, kde=True)
    plt.xlabel("VADER Compound Score")
    plt.title("VADER Compound Score Distribution by Label")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualization 3: KDE plot comparison (TextBlob polarity vs VADER compound) ---
    print("📊 KDE comparison between fake and real for both methods...")
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x="tb_polarity", hue="label", fill=True, common_norm=False, palette="Set2")
    plt.xlabel("TextBlob Polarity")
    plt.title("TextBlob Sentiment Comparison (Fake vs Real)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x="vader_compound", hue="label", fill=True, common_norm=False, palette="Set1")
    plt.xlabel("VADER Compound Score")
    plt.title("VADER Sentiment Comparison (Fake vs Real)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualization 4: Scatter comparison of both methods ---
    print("📊 Scatter: TextBlob vs VADER...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="tb_polarity",
        y="vader_compound",
        hue="label",
        alpha=0.5,
        palette={0: "red", 1: "blue"},
    )
    plt.title("Sentiment Comparison: TextBlob Polarity vs VADER Compound")
    plt.xlabel("TextBlob Polarity")
    plt.ylabel("VADER Compound")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Compute KS-statistic and AUC for both sentiment methods ---
    print("\n📈 Statistical Comparison Between Fake (0) and Real (1) Articles:")

    # --- TEXTBLOB POLARITY ---
    ks_tb = ks_2samp(df[df["label"] == 0]["tb_polarity"], df[df["label"] == 1]["tb_polarity"])
    auc_tb = roc_auc_score(df["label"], df["tb_polarity"])
    print(f"🔹 TextBlob - KS statistic: {ks_tb.statistic:.4f} (p={ks_tb.pvalue:.4f}), AUC: {auc_tb:.4f}")

    # --- VADER COMPOUND ---
    ks_vader = ks_2samp(df[df["label"] == 0]["vader_compound"], df[df["label"] == 1]["vader_compound"])
    auc_vader = roc_auc_score(df["label"], df["vader_compound"])
    print(f"🔸 VADER    - KS statistic: {ks_vader.statistic:.4f} (p={ks_vader.pvalue:.4f}), AUC: {auc_vader:.4f}")

    # Save results in csv file
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        try:
            df.to_csv(save_csv_path, index=False)
            print(f"✅ Sentiment-labeled dataset saved to: {save_csv_path}")
        except PermissionError:
            print(f"❌ Cannot write to {save_csv_path}. Is the file open or write-protected?")

    # Save results in mongoDB articles table...
    '''
    if save_in_mongoDB:
        # Update MongoDB articles with sentiment scores
        print("📝 Updating MongoDB with sentiment scores...")
        # for _, row in df.iterrows():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Updating Articles"):
            update_fields = {
                "polarity": row["tb_polarity"],
                "subjectivity": row["tb_subjectivity"],
                "v_compound": row["vader_compound"],
                "sentiment": "positive" if row["vader_compound"] >= 0.05 else "negative" if row["vader_compound"] <= -0.05 else "neutral"
            }
            articles_col.update_one({"_id": row["_id"]}, {"$set": update_fields})
        print("✅ Sentiment scores saved to MongoDB.")
    '''
    if save_in_mongoDB:
        from pymongo import UpdateOne

        print("📝 Updating MongoDB with sentiment scores in batches...")
        operations = []
        batch_size = 200  # You can tweak this size for performance

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Updating Articles"):
            update_fields = {
                "polarity": row["tb_polarity"],
                "subjectivity": row["tb_subjectivity"],
                "v_compound": row["vader_compound"],
                "sentiment": (
                    "positive" if row["vader_compound"] >= 0.05
                    else "negative" if row["vader_compound"] <= -0.05
                    else "neutral"
                )
            }
            operations.append(UpdateOne({"_id": row["_id"]}, {"$set": update_fields}))

            if len(operations) >= batch_size:
                articles_col.bulk_write(operations)
                operations = []

        # Final leftover operations (sentiment indexes) to save in mongodb articles
        if operations:
            articles_col.bulk_write(operations)

        print("✅ Sentiment scores saved to MongoDB.")

        # Grouped Sentiment Statistics by Subject...
        if "subject" in df.columns:
            print("\n📊 Sentiment Statistics by Subject:\n")
            stats_by_subject = df.groupby("subject")[["tb_polarity", "tb_subjectivity", "vader_compound"]].mean().round(3)
            print(stats_by_subject)

            stats_by_subject.to_csv("./fakenews_datasets/sentiment/sentiment_by_subject.csv")

        return df
