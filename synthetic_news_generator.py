import requests
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
from tqdm import tqdm
import requests
import time
from sentence_transformers import SentenceTransformer # For sentence embeddings   --> pip install -U sentence-transformers
from tqdm import tqdm


# Enable progress_apply for pandas
tqdm.pandas()


def fetch_real_news(api_key, query, n_articles=95):
    print(f"🔍 Fetching real news for query: '{query}'...")
    articles = []
    page = 1
    while len(articles) < n_articles and page <= 1:  # Fetch multiple pages if needed
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&pageSize=100&page={page}&language=en&sortBy=relevancy&apiKey={api_key}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ API error {response.status_code}: {response.text}")
            break
        data = response.json()
        page_articles = data.get("articles", [])
        if not page_articles:
            break
        articles.extend(page_articles)
        page += 1
        time.sleep(1)  # Respect API rate limits

    news_data = []
    for article in articles[:n_articles]:

        title = article.get("title") or ""
        text = article.get("description") or ""
        title = title.strip()
        text = text.strip()

        # title = article.get("title", "").strip()
        # text = article.get("description", "").strip()

        if not title or not text:
            continue
        date = article.get("publishedAt", datetime.today().strftime('%B %d, %Y'))
        news_data.append({
            "title": title[:200],
            "text": text[:1000],
            "subject": query,
            "date": date,
            "label": 1
        })
    print(f"✅ Fetched {len(news_data)} real articles.\n")
    print(pd.DataFrame(news_data).head(90))

    return news_data


def generate_fake_news_from_real(real_news_data, generator):
    print("🧠 Generating fake news...")
    fake_news_data = []

    for real_article in tqdm(real_news_data, desc="Generating fake news"):
        prompt = real_article['title']  # just the title, no extra text
        try:
            output = generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
        except Exception as e:
            print(f"⚠️ Skipping one article due to error: {e}")
            continue

        # Strip the prompt from the generated text (if echoed) and clean up
        fake_output = output.replace(prompt, "").strip()
        fake_lines = fake_output.split('\n')

        # Extract title and text from the generated output
        fake_title = fake_lines[0][:100] if fake_lines else "Generated Fake Title"
        fake_text = ' '.join(fake_lines[1:]).strip()[:300]

        # Fallback if text is too short
        if len(fake_text) < 30:
            fake_text += " (This article was auto-generated.)"

        fake_date = (datetime.today() - timedelta(days=random.randint(0, 1500))).strftime('%B %d, %Y')

        fake_news_data.append({
            "title": fake_title,
            "text": fake_text,
            "subject": real_article['subject'],  # make sure subject is set properly
            "date": fake_date,
            "label": 0
        })

    print(f"✅ Generated {len(fake_news_data)} fake articles.")
    print(pd.DataFrame(fake_news_data).head(90))

    return fake_news_data


# GPT2 model and NewsAPI are used to create_semi_synthetic_news_dataset. Generator is created by
# Hugging Face's Transformers library
# API key from https://newsapi.org/ is required !!!
def create_semi_synthetic_news_dataset(api_key, query="technology", save_path=None, n_per_class=95, seed=42):
    random.seed(seed)
    generator = pipeline("text-generation", model="gpt2")

    real_news_data = fetch_real_news(api_key, query, n_articles=n_per_class)
    if not real_news_data:
        print("❌ No real news to use.")
        return None

    fake_news_data = generate_fake_news_from_real(real_news_data, generator)
    all_data = real_news_data + fake_news_data
    df = pd.DataFrame(all_data).sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(save_path, index=False, quoting=1)  # quoting=1 adds quotes around fields with commas
    print(f"\n✅ Dataset saved to: {save_path} ({len(df)} rows)")
    return df


def import_synthetic_other_csv_to_mongodb(filepath):
    from pymongo import MongoClient
    from datetime import datetime
    import pandas as pd
    import random

    client = MongoClient("mongodb://localhost:27017/")
    db = client["realtime_news_db"]
    articles_col = db["articles"]

    df = pd.read_csv(filepath)
    count = 0
    count_empty = 0
    count_nolabel = 0
    invalid_date_count = 0
    # for _, row in df.iterrows():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Updating Articles"):

        # some controls
        if not row["label"]:
            count_nolabel +=1
        if not (row["title"] and row["text"]):
            count_empty += 1
        else:
            date_raw = pd.to_datetime(row.get("date"), errors="coerce")
            # article_date = date_raw if pd.notnull(date_raw) else datetime.today()
            if pd.isnull(date_raw):
                invalid_date_count += 1
                date_raw = datetime.today()

            article = {
                "title": row["title"][:200],
                "text": row["text"][:1000],
                # read row's content of "subject" field & if not found --> "General"
                "subject": row.get("subject", "General"),
                # example : row = {"title": "AI is changing healthcare", "text": "..."} row.get("subject", "General") → returns "General"
                # "date": datetime.strptime(row.get("date", datetime.today().strftime("%Y-%m-%d")), "%Y-%m-%d"),
                "date": date_raw,
                # following line safely sets the label field in each MongoDB article  from the "label" field from current row
                # of the CSV file. if label field missing or None, it defaults to 0. saved in integer format.
                "label": int(row.get("label", 0)),
                #"confidence": 0.12,
                "polarity": 0,
                "subjectivity": 0,
                "v_compound": 0,
                "sentiment": "neutral",
                #"score": -0.3,
                "source": "Synthetic ",
                "submitted_by": "NewsAPI+GPT2",
                "reviewed": False,
                "comments": []
            }
            articles_col.insert_one(article)
            count += 1
    # some statistics on the synthetic results...
    total=len(df)
    print("total articles= {total} no labeled= {count_nolabel}  empty (no titles/text)={count_empty}")
    return count


def import_synthetic_other_csv_to_mongodb_new(filepath):
    from pymongo import MongoClient
    from datetime import datetime
    import pandas as pd
    import random
    from tqdm import tqdm

    client = MongoClient("mongodb://localhost:27017/")
    db = client["realtime_news_db"]
    articles_col = db["articles"]

    df = pd.read_csv(filepath, keep_default_na=False)
    count_inserted = 0
    count_empty = 0
    count_nolabel = 0
    invalid_date_count = 0
    count_rejected_short_or_symbol = 0
    fixed_title_from_text = 0

    rejected_articles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading Synthetic Articles"):

        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        label = pd.to_numeric(row.get("label"), errors="coerce")

        # Initialize rejection flags
        reject = False
        this_row_invalid = False

        # Basic validation
        if pd.isna(label) or label not in (0, 1):
            count_nolabel += 1
            reject = True

        # if  title or  text are empty
        if not title or not text:
            count_empty += 1
            reject = True

        if (not title or len(title) < 10) and text and len(text) >= 25:
            title = text[:40]
            title_fixed = True
            fixed_title_from_text += 1
        else:
            title_fixed = False

        # if very small length of title or text
        if len(title) < 10 or len(text) < 25:
            count_rejected_short_or_symbol += 1
            reject = True

        # Reject articles with very short or symbol-only content
        symbols_only = not any(char.isalnum() for char in title) or not any(char.isalnum() for char in text)
        if symbols_only:
            count_rejected_short_or_symbol += 1
            reject = True

        # if any of the above controls is valid reject the article...
        if reject:
            rejected_articles.append({
                "title": title,
                "text": text[:120],
                "label": label,
                "reasons": {
                    "invalid_label": pd.isna(label) or label not in (0, 1),
                    "empty_title_or_text": not title or not text,
                    "too_short": len(title) < 10 or len(text) < 25,
                    "symbols_only": symbols_only
                }
            })
            continue    # Skip article after counting all reasons

        # Parse date
        date_raw = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isnull(date_raw):
            invalid_date_count += 1
            date_raw = datetime.today()

        article = {
            "title": title[:200],
            "text": text[:1000],
            "subject": row.get("subject", "General"),
            "date": date_raw,
            "label": int(label),
            "polarity": 0,
            "subjectivity": 0,
            "v_compound": 0,
            "sentiment": "neutral",
            "source": "Synthetic",
            "submitted_by": "NewsAPI+GPT2",
            "reviewed": False,
            "comments": [],
            "title_fixed": title_fixed  # new field
        }

        articles_col.insert_one(article)
        count_inserted += 1

    # Print stats
    total = len(df)
    print(f"📄 Total rows processed: {total}")
    print(f"✅ Inserted: {count_inserted}")
    print(f"⚠️  Skipped (no label): {count_nolabel}")
    print(f"⚠️  Skipped (empty title/text): {count_empty}")
    print(f"⚠️  Skipped (too short or symbols only): {count_rejected_short_or_symbol}")
    print(f"🕒 Defaulted today's date for {invalid_date_count} rows")
    print(f"🛠 Titles auto-filled from text: {fixed_title_from_text}")  # ✅ ADD THIS LINE

    print("\n🛑 Sample of rejected articles:")

    for i, art in enumerate(rejected_articles[:15], start=1):
        print(f"\n#{i}")
        print(f"Title: {art['title']}")
        print(f"Text: {art['text']}")
        print(f"Label: {art['label']}")
        print(f"Reasons: {art['reasons']}")

    return count_inserted
