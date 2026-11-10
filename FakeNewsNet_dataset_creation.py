 #  Loads political FakeNewsNet CSVs, cleans them, generates fake dates,
 #  and saves three datasets: Fake1.csv, True1.csv, AllNews_FakeNewsNet.csv.
#
import os
import pandas as pd
import re
import random
from datetime import datetime, timedelta
from bs4 import BeautifulSoup   # pip install beautifulsoup4

def Create_fakenewsnet_dataset_from_files(
    input_dir="./fakenews_datasets/FakeNewsNet",
    output_dir=None,
    fake_files=["politifact_fake.csv", "gossipcop_fake.csv"],
    real_files=["politifact_real.csv", "gossipcop_real.csv"],
    OutputFileName="AllNews_FakeNewsNet_1.csv",
    verbose=True):


    if output_dir is None:
        output_dir = input_dir

    def generate_random_date(start_year=2016, end_year=2020):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        delta = end_date - start_date
        return (start_date + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

    def clean_text(text):
        if pd.isna(text):
            return ""
        try:
            if "<" in text and ">" in text:
                text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            pass
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def is_junk_text(text):
        return (
            pd.isna(text)
            or text.strip() == ""
            or re.fullmatch(r"[\d\s\t,]+", text) is not None
        )

    def process_file(file_path, label, source_name):
        df = pd.read_csv(file_path)

        if 'title' not in df.columns:
            raise ValueError(f"'title' column not found in {file_path}")

        df["title"] = df["title"].fillna("")
        df["text"] = df["title"]  # fallback for missing 'text'

        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)
        df = df[~df["text"].apply(is_junk_text)]

        df["label"] = label
        df["subject"] = source_name
        df["date"] = [generate_random_date() for _ in range(len(df))]

        return df[["title", "text", "subject", "date", "label"]]

    # Process fake
    df_fake = pd.concat([
        process_file(os.path.join(input_dir, fname), label=0, source_name="politifact" if "politifact" in fname else "gossipcop")
        for fname in fake_files
    ], ignore_index=True).drop_duplicates(subset=["title", "text"])

    # Process real
    df_real = pd.concat([
        process_file(os.path.join(input_dir, fname), label=1, source_name="politifact" if "politifact" in fname else "gossipcop")
        for fname in real_files
    ], ignore_index=True).drop_duplicates(subset=["title", "text"])

    # Save datasets
    fake_path = os.path.join(output_dir, "Fake1.csv")
    real_path = os.path.join(output_dir, "True1.csv")
    # combined_path = os.path.join(output_dir, "AllNews_FakeNewsNet.csv")
    combined_path = os.path.join(output_dir, OutputFileName)  #
    df_fake.to_csv(fake_path, index=False)
    df_real.to_csv(real_path, index=False)
    pd.concat([df_fake, df_real], ignore_index=True).sample(frac=1, random_state=42).to_csv(combined_path, index=False)

    if verbose:
        print(f"✅ Saved:\n- {fake_path} ({len(df_fake)} fake)\n- {real_path} ({len(df_real)} real)\n- {combined_path} total")

    return df_fake, df_real


if __name__ == "__main__":
    Create_fakenewsnet_dataset_from_files()
