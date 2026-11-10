# Transformer inference for the desktop application.
import tkinter as tk
from pymongo import MongoClient
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ====== GLOBAL CACHE
_bert_classifier = None
_flan_classifier = None
_flan_tokenizer = None
_bart_classifier = None
_roberta_classifier = None
'''
# model="mariagrandury/roberta-base-fake-news NOT EXIST ANY MORE IN HUGGING FACE !!!
# REPLACED BY THE NEXT MODEL WWITH THE SAME NAME
# roberta is similar to bert... Model: mariagrandury/roberta-base-fake-news
# Finetuned on fake/real news detection
# Very good balance between accuracy and speed
def classify_with_roberta(article_text):
    """
    Classify using mariagrandury/roberta-base-fake-news model.
    """
    global _roberta_classifier
    if _roberta_classifier is None:
        _roberta_classifier = pipeline(
            "text-classification",
            model="mariagrandury/roberta-base-fake-news",
            tokenizer="mariagrandury/roberta-base-fake-news",
            truncation=True,
            max_length=512
        )

    result = _roberta_classifier(article_text)[0]
    label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}
    label = label_map.get(result["label"], "Unknown")
    confidence = result["score"]
    return {"label": label, "confidence": confidence}
'''

def classify_with_roberta(article_text):
    """
    Classify using hamzab/roberta-fake-news-classification.
    This model expects format: <title> TITLE HERE <content> CONTENT HERE <end>
    """
    global _roberta_classifier
    if _roberta_classifier is None:
        _roberta_classifier = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
            tokenizer="hamzab/roberta-fake-news-classification",
            truncation=True,
            max_length=512
        )

    # You can structure input as per model's instructions
    input_text = f"<title> {article_text} <end>"
    result = _roberta_classifier(input_text)[0]
    label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}
    label = label_map.get(result["label"], result["label"])
    confidence = result["score"]
    return {"label": label, "confidence": confidence}




def classify_with_bart(article_text):
    """
    Classifies a given article as Fake or Real using facebook/bart-base model.
    Returns a dictionary with label and confidence. model_id4 = "facebook/bart-base"
    developed by Facebook AI. combines the best parts of BERT
    (good at understanding text) and GPT (good at generating text)
    It can be used for classification, summarization, translation, and text generation.
    facebook/bart-base is smaller (than full-size BART) but very strong.
    It is encoder-decoder architecture (like T5).
    Pre-trained on masking parts of sentences and reconstructing them, so it understands structure very well.
    In classification, it usually works better than tiny BERT, but worse than very large models (unless fine-tuned).
    139 million parameters (bigger than bert-tiny. Speed	: Fair (faster than big LLMs, slower than bert-tiny)
    No HuggingFace token is needed (it’s public).

   IMPORTANT : facebook/bart-base is not specially fine-tuned for Fake News, so its predictions may be less accurate
   unless you fine-tune it.
    """
    global _bart_classifier
    if _bart_classifier is None:
        _bart_classifier = pipeline(
            "text-classification",
            model="facebook/bart-base",
            tokenizer="facebook/bart-base",
            truncation=True,
            max_length=512
        )

    result = _bart_classifier(article_text)[0]
    label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}  # Adjust if necessary depending on output
    label = label_map.get(result["label"], result["label"])  # Fallback: if unknown label, keep it
    confidence = result["score"]

    return {"label": label, "confidence": confidence}


# BERT is a lightweight pretrained Bidirectional transformer model not an LLM model in the modern sense.
# (for Text classification, Question Answer, Sentiment analysis)
#
#  Feature	        BERT	                                         Modern LLMs (like GPT, LLaMA, Claude)
#  ----------------------------------------------------------------------------------------------------------
#  Direction	    Bidirectional (reads both sides)	             Autoregressive (generates left-to-right)
#  Primary Use	    Understanding tasks (e.g. classification, Q&A)	 Generation (e.g. chat, code, summarization)
#  Size (usually)	Small to Medium (110M–300M)	                     Large (billions of parameters)
#  Output Type	    Not generative (no text generation)	             Generative
#  Example Models	BERT, DistilBERT, TinyBERT	                     GPT-3, GPT-4, LLaMA, Claude, PaLM

def classify_with_mrm8488_bert(article_text):
    """
    Classify using the mrm8488/bert-tiny-finetuned-fake-news-detection model.
    """
    global _bert_classifier
    if _bert_classifier is None:
        _bert_classifier = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-fake-news-detection",
            tokenizer="mrm8488/bert-tiny-finetuned-fake-news-detection",
            truncation=True,  # Enable truncation
            max_length=512    # cut the article text to first 512 chars
        )

    result = _bert_classifier(article_text)[0]
    label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}
    label = label_map.get(result["label"], "Unknown")
    confidence = result["score"]
    return {"label": label, "confidence": confidence}

    # call : classification_result = classify_article_with_bert(example_article)
    #        print("📰 Classification:", classification_result["label"], f"({classification_result['score']:.2f} confidence)")
    # output --> Classification: REAL (0.87 confidence)
    # article = "The moon is made of cheese, according to scientists at NASA."
    #
    # prediction = classify_article_with_bert(article)
    # print(f"🧠 Prediction: {prediction['label']} (confidence: {prediction['score']:.2f})")

# more general LLM model produces next word in a given prompt...
def classify_with_llm_flan_t5(article_text, load_path="./fakenews_datasets/models/flan_t5_base"): # load_path not used here...
    """
    Classify using Hugging Face's google/flan-t5-base model.
    Downloads the model automatically if not already cached.
    PREVIOUS FUNCTION USES THE LOCAL DIR TO GET THE MODEL.
    """
    global _flan_classifier, _flan_tokenizer
    if _flan_classifier is None or _flan_tokenizer is None:
        model_name = "google/flan-t5-base"
        # model_name = load_path  if I want to use the model from the local directory...
        _flan_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _flan_classifier = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prompt = f"Is the following article fake news? Just say 'Fake' or 'Real': {article_text}"
    inputs = _flan_tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = _flan_classifier.generate(**inputs, max_new_tokens=3, num_beams=2, do_sample=False) # max_new_tokens=20 --> for more words after the prompt...
    result = _flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"label": result.strip(), "confidence": 1.0}  # Flan doesn't output confidence



def classif_model_selection_popup(root):
    popup = tk.Toplevel(root)
    popup.title("Select LLM Model")
    popup.geometry("300x250")

    tk.Label(popup, text="Choose a model for classification:", font=("Arial", 10, "bold")).pack(pady=10)

    model_var = tk.StringVar(value="bert")
    use_full_text_var = tk.BooleanVar(value=False)  # Checkbox variable

    # Radio buttons for model selection
    models = [
    ("BERT (mrm8488)", "bert"),
    ("FLAN-T5", "flan"),
    ("BART-base", "bart"),
    ("RoBERTa (hamzab)", "roberta")
    ]
    for label, value in models:
        tk.Radiobutton(popup, text=label, variable=model_var, value=value).pack(anchor="w", padx=20)

    # Add checkbox for full text usage
    tk.Checkbutton(popup, text="Use full text (title + body)", variable=use_full_text_var).pack(pady=10)

    def on_confirm():
        model_type = model_var.get()
        use_full_text = use_full_text_var.get()
        popup.destroy()
        classify_all_articles(model_type=model_type, use_full_text=use_full_text)

    tk.Button(popup, text="Start Classification", command=on_confirm).pack(pady=10)



def classify_all_articles(model_type="bert", use_full_text=False):
    """
    Classify all articles in MongoDB using the selected model ('bert' or 'flan-t5').
    """
    from tqdm import tqdm
    import pymongo

    client = MongoClient("mongodb://localhost:27017/")
    db = client["realtime_news_db"]
    articles_col = db["articles"]
    n_articles = articles_col.count_documents({})
    print(f"🔄 Classifying {n_articles} articles using {model_type.upper()} model...")

    cursor = articles_col.find(
        {"text": {"$ne": None}, "label": {"$in": [0, 1]}},
        no_cursor_timeout=False  # Atlas tier doesn't allow True
    ).batch_size(200)

    correct, total, errors, success, fail = 0, 0, 0, 0, 0
    y_true, y_pred = [], []
    bulk_updates = []
    batch_size = 200

    # cursor = articles_col.find().batch_size(100)
    try:
        for article in tqdm(cursor, total=n_articles): # try: for article in tqdm(articles_col.find(), total=n_articles):
            try:
                article_id = article["_id"]
                # remark for this change --> total article gives low quality classif so only title for now...
                # if use_full_text: text = article.get("title", "") + " " + article.get("text", "")
                # else: text = article.get("title", "")
                title = article.get("title", "")
                body = article.get("text", "")
                text = f"{title} {body}" if use_full_text else title
                # if text empty ignore it
                if not text:
                    continue
                if model_type == "bert":
                    prediction = classify_with_mrm8488_bert(text)
                elif model_type == "flan":
                    prediction = classify_with_llm_flan_t5(text)
                elif model_type == "bart":
                    prediction = classify_with_bart(text)
                elif model_type == "roberta":
                    prediction = classify_with_roberta(text)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                # Save new fields. Instead of the next line I replaced it with the following block for batch update...
                # articles_col.update_one( {"_id": article_id}, {"$set": {"llm_label": prediction["label"], "llm_confidence": prediction["confidence"]}} )
                bulk_updates.append(
                    pymongo.UpdateOne(
                        {"_id": article_id},
                        {"$set": {
                            "llm_label": prediction["label"],
                            "llm_confidence": prediction["confidence"]
                        }}
                    )
                )
                if len(bulk_updates) >= batch_size:
                    articles_col.bulk_write(bulk_updates)
                    bulk_updates = []

                # Compare with existing label (if label exists)
                if "label" in article:
                    expected = "Real" if article["label"] == 1 else "Fake"
                    predicted = prediction["label"]

                    y_true.append(expected)
                    y_pred.append(predicted)

                    if expected.lower() == predicted.lower():
                        correct += 1
                    total += 1

                success += 1

            except Exception as e:
                print(f"❌ Error with article ID {article['_id']}: {e}")
                fail += 1
                errors += 1

        # Write any remaining updates...
        if bulk_updates:
            articles_col.bulk_write(bulk_updates)
    finally:
        cursor.close()


    # --- Final evaluation ---
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n📊 Classification Results:")
    print(f" - Total articles checked: {total}")
    print(f" - Correct classifications: {correct}")
    print(f" - Errors: {errors}")
    print(f" - Overall Accuracy: {accuracy:.2f}%")

    # --- Plotting Confusion Matrix ---
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred, labels=["Fake", "Real"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        plt.title(f"Confusion Matrix ({model_type.upper()} Classification)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # --- Save evaluation results to MongoDB ---
    evaluation = {
        "model_type": model_type,
        "accuracy": accuracy,
        "total_articles": total,
        "correct_classifications": correct,
        "errors": errors,
        "timestamp": datetime.utcnow(),  # Save time of evaluation
    }
    db["classification_results"].insert_one(evaluation)

    # To see past evaluations from mongodb...
    # for doc in db["classification_results"].find().sort("timestamp", -1): print(doc)

    return accuracy
