# Fake News Detection

A University project of mine about exploring how classical machine learning and pretrained transformers classify labeled news articles. A Tkinter desktop app uses MongoDB to browse and filter articles, train classifiers on the current selection, and run transformer inference. The main workflow focuses on English-language news.

## What the project covers

| Component | Implementation |
| --- | --- |
| Data ingestion | Import labeled CSVs into MongoDB |
| Classical ML | Compare logistic regression, random forest, and XGBoost with TF-IDF. Compare logistic regression and random forest with Sentence-BERT embeddings |
| Evaluation | Stratified 80/20 holdout, five-fold stratified cross-validation, confusion matrices, F1, and ROC AUC |
| Transformer inference | Run pretrained BERT, BART, RoBERTa, or prompt-based FLAN-T5 on individual articles or a collection |
| Dataset generation | Generate semi-synthetic English articles with NewsAPI and GPT-2 |
| Sentiment exploration | Inspect TextBlob and VADER scores alongside article labels |

## Run the desktop demo

Use Python 3.11, Tkinter, and a local MongoDB server listening on `127.0.0.1:27017`. Use a disposable local MongoDB instance for the demo. The application uses the `realtime_news_db` database and seeds a local demo login.

Run these commands from the folder containing `requirements.txt`. If you downloaded the ZIP from GitHub, extract it first. Only `setupDB.py` and `setup_realtime_news_db_1.py` are launched directly for the desktop demo.

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in Bash. Then run:

```sh
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords vader_lexicon
python setupDB.py
python setup_realtime_news_db_1.py
```

1. Sign in as `admin@example.test` with password `local-demo`.
2. Use **Upload Synthetic/other dataset** and select [examples/demo_articles.csv](examples/demo_articles.csv).
3. Use **Train Models from MongoDB Articles** and choose **TF-IDF**. Close the metrics and ROC plot windows to return to the app.
4. To try pretrained inference, select or enter an article title and click **LLM Classify**, choose a model, and click **Instant Classify Article**. The first use downloads the selected model. Sentence-BERT training also downloads a model on first use.

The example CSV contains invented English and Greek text, with ten rows per class. Its labels are for exercising ingestion and training. Any scores from this sample are smoke-test output. The **Reset & Load Kaggle Data** button clears the demo collections after confirmation.

## Use your own data

- **CSV import.** Supply UTF-8 `title`, `text`, and `label` columns. Use `0` for fake and `1` for real. You can also provide `subject` and `date`. The GUI applies minimum text lengths and expects at least five usable rows per class to train.
- **Kaggle.** The original coursework used the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Combine the source CSVs with a binary `label` column and use the CSV import, or put the combined file at `fakenews_datasets/Kaggle/AllNews_Kaggle.csv` for the app's reset-and-load action.
- **Optional generation.** Set `NEWSAPI_KEY` before launching the desktop app for English generation. This requires network access. The original datasets and model artifacts are kept outside Git.

## Interpreting the output

The evaluation measures agreement with dataset labels on a random split. Publisher patterns, duplicate coverage, and generated text can make these scores optimistic for unseen sources. The coursework uses English-oriented preprocessing and sentiment tools. The demo checks that Greek text can be imported. The coursework did not establish Greek classification accuracy. Treat predictions as model outputs. Verifying a real-world claim requires independent evidence.

## Credits

Nikolas Kousoulis. 
This was a project I created while attending the course 'Advanced Data Management' of the Department of Electrical and Computer Engineering.

## License

See [LICENSE](LICENSE) for the MIT license. Third-party datasets, models, and dependencies have their own terms.
