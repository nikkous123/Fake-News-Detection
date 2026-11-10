import os
import requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
# pip install protobuf tiktoken
# pip install sentencepiece
# === CONFIGURATION ===
MODEL_NAME = "google/mt5-small"
NEWS_LIMIT = 100

# Greek language model by nlpaueb on Hugging Face is:
# nlpaueb/bert-base-greek-uncased-v1
# This is a BERT-based model pretrained on Greek text and supports text classification,
# sentence embedding, and feature extraction. It is not a generative model like T5 or GPT,
# so it cannot be used to generate fake Greek news.

# === Load model & tokenizer ===

# === Connect to MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["realtime_news_db"]
articles_col = db["articles"]

def fetch_greek_real_articles(api_key, max_articles=NEWS_LIMIT):
    articles = []
    params = {"apikey": api_key, "country": "gr", "language": "el", "q": "τεχνολογία"}
    seen_pages = set()
    while len(articles) < max_articles:
        response = requests.get("https://newsdata.io/api/1/news", params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Newsdata request failed with HTTP {response.status_code}.")
        data = response.json()
        results = data.get("results")
        if not isinstance(results, list) or not results:
            break

        for item in results:
            if isinstance(item, dict) and item.get("title") and item.get("content"):
                articles.append({
                    "title": item["title"],
                    "text": item["content"],
                    "date": item.get("pubDate"),
                    "subject": item.get("category") or "unknown",
                })
                if len(articles) >= max_articles:
                    break

        next_page = data.get("nextPage")
        if not next_page or next_page in seen_pages:
            break
        seen_pages.add(next_page)
        params["page"] = next_page

    return pd.DataFrame(articles)


def generate_fake_article(prompt_title, tokenizer, model):
    prompt = f"Δημιούργησε ένα ψεύτικο άρθρο βασισμένο στον τίτλο: {prompt_title}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output_ids = model.generate(input_ids, max_length=256, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def insert_articles_to_mongodb(df: pd.DataFrame, label: int, source: str):
    batch = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting {source}"):
        date = pd.to_datetime(row["date"], errors="coerce", utc=True)
        article = {
            "title": row["title"],
            "text": row["text"],
            "subject": row.get("subject", ""),
            "date": datetime.utcnow() if pd.isna(date) else date.to_pydatetime(),
            "label": label,  # 1 = Real, 0 = Fake
            "polarity": 0,
            "subjectivity": 0,
            "v_compound": 0,
            "sentiment": "neutral",
            "source": source,
            "submitted_by": None,
            "reviewed": False,
            "comments": []
        }
        batch.append(article)
    if batch:
        articles_col.insert_many(batch)

def generate_and_store_greek_fake_news(api_key):
    print("📥 Fetching Greek real articles...")
    real_df = fetch_greek_real_articles(api_key)
    if real_df.empty:
        print("No articles with usable text were returned.")
        return
    print(f"✅ {len(real_df)} real articles fetched.")

    print("🤖 Generating fake articles...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    fake_texts = [
        generate_fake_article(title, tokenizer, model)
        for title in tqdm(real_df["title"], desc="Generating Fakes")
    ]
    fake_df = real_df.copy()
    fake_df["text"] = fake_texts

    print("💾 Saving to MongoDB...")
    insert_articles_to_mongodb(real_df, label=1, source="Newsdata (Greek)")
    insert_articles_to_mongodb(fake_df, label=0, source="GPT-MT5 (Greek Synthetic)")
    print("✅ Articles saved.")

if __name__ == "__main__":
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        raise SystemExit("Set NEWSDATA_API_KEY before running this script.")
    generate_and_store_greek_fake_news(api_key)
