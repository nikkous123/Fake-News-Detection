import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
from pymongo import MongoClient
from datetime import datetime
import uuid
import csv
import os
import random
import pandas as pd
from tqdm import tqdm

from synthetic_news_generator import create_semi_synthetic_news_dataset, import_synthetic_other_csv_to_mongodb_new
from TrainModels_from_dataset import Train_Models_from_dataframe
from Sentiment_Analysis_on_articles import sentiment_analysis_on_articles
from classify_all_articles1 import (classify_with_mrm8488_bert,
                                    classify_with_llm_flan_t5,
                                    classify_with_bart,
                                    classify_with_roberta,
                                    classif_model_selection_popup)

# working directory : mongoDB_material
# environment : mongodb_env, python 3.11
# see Hugin face login, token, models in classify_all_articles1.py
# --- MongoDB Connection ---
client = MongoClient("mongodb://localhost:27017/")
db = client["realtime_news_db"]
articles_col = db["articles"]
users_col = db["users"]
subjects_col = db["subjects"]
user_logs_col = db["user_logs"]


root = tk.Tk()
root.withdraw()  # Hide main window until login

def create_admin_buttons():
    admin_button_frame = tk.Frame(articles_tab)
    admin_button_frame.pack(pady=5)
    tk.Button(admin_button_frame, text="Generate Synthetic Dataset", command=lambda: generate_synthetic_dataset(api_key, selected_subject_var.get())).pack(side="left", padx=5)
    tk.Button(admin_button_frame, text="Import Synthetic (or other) csv dataset into MongoDB", command=lambda: run_csv_import()).pack(side="left", padx=5)



from datetime import datetime

def log_user_action(action_type, description=""):
    if current_user:
        try:
            log_entry = {
                "user_email": current_user.get("email", "unknown"),
                "timestamp": datetime.now(),
                "action": action_type,
                "description": description
            }
            user_logs_col.insert_one(log_entry)
            print(f"📜 Logged action: {action_type} | {description}")

        except Exception as e:
            print(f"⚠️ Failed to log action: {e}")


current_user = {}

def show_login_window():
    login_win = tk.Toplevel()
    login_win.title("Login")
    login_win.geometry("300x180")
    login_win.grab_set()

    def cancel_login():
        root.destroy()
    login_win.protocol("WM_DELETE_WINDOW", cancel_login)

    tk.Label(login_win, text="Email:").pack(pady=5)
    email_entry = tk.Entry(login_win, width=30)
    email_entry.pack()

    tk.Label(login_win, text="Password:").pack(pady=5)
    password_entry = tk.Entry(login_win, show="*", width=30)
    password_entry.pack()

    def do_login():
        email = email_entry.get().strip()
        password = password_entry.get().strip()
        user = users_col.find_one({"email": email, "password": password})
        if user:
            global current_user
            current_user = user
            log_user_action("login", f"User {current_user['email']} logged in.")
            login_win.destroy()
            root.deiconify()  # Show main app window
            create_admin_buttons()
            '''
            # create Delete button after login, using lambda to avoid function ordering issues
            delete_button_frame = tk.Frame(articles_tab)
            delete_button = tk.Button(delete_button_frame, text="🗑 Delete Selected Articles", command=lambda: delete_selected_articles())
            delete_button.pack()
            delete_button_frame.pack(pady=5)
            '''
        else:
            messagebox.showerror("Login Failed", "Invalid email or password.")

    tk.Button(login_win, text="Login", command=do_login).pack(pady=10)

show_login_window()


# when double click on an article from the grid, copy it to the search field above grid...
def on_article_double_click(event):
    selected = articles_tree.focus()
    if selected:
        values = articles_tree.item(selected)["values"]
        title = values[1]  # assuming title is in column index 1
        search_entry.delete(0, tk.END)
        search_entry.insert(0, title)


# calls function import_synthetic_csv_to_mongodb() that uploads a dataset selected from the file explorer...
def run_csv_import():
    filepath = filedialog.askopenfilename(title="Select Synthetic (or other) CSV File", filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return
    try:
        count = import_synthetic_other_csv_to_mongodb_new(filepath)
        messagebox.showinfo("Import Success", f"✅ Imported {count} articles from CSV.")
        refresh_articles()
        log_user_action("import_synthetic_other_csv_to_mongodb", f"Imported synthetic dataset from {filepath}")

    except Exception as e:
        messagebox.showerror("Import Failed", f"❌ {e}")


# --- GUI Setup ---

root.title("📰 Realtime News Monitor")
root.geometry("1200x700")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# === Articles Tab ===
articles_tab = ttk.Frame(notebook)
notebook.add(articles_tab, text="Articles")

# Search Fields
search_frame = tk.Frame(articles_tab)
search_frame.pack(pady=5)

search_label = tk.Label(search_frame, text="Search:")
search_label.pack(side="left")

search_field_var = tk.StringVar()
search_field = ttk.Combobox(search_frame, textvariable=search_field_var, values=["title", "subject", "sentiment", "source", "label", "confidence"], state="readonly", width=12)
search_field.set("title")  # default
search_field.pack(side="left", padx=5)
search_entry = tk.Entry(search_frame, width=40)
search_entry.pack(side="left", padx=5)

llm_result_var = tk.StringVar()
llm_result_label = tk.Label(search_frame, textvariable=llm_result_var, font=("Arial", 10, "bold"), fg="blue")
llm_result_label.pack(side="left", padx=10)


# Reset & Load Kaggle Data button
tk.Button(articles_tab, text="Reset & Load Kaggle Data", command=lambda: setup_sample_data()).pack(pady=5)
# classify all articles   e.g. model_type="flan-t5" ????
# tk.Button(articles_tab, text="Classify all articles with model :...", command=lambda: classify_all_articles(model_type="bert")).pack(pady=5)
# Replace old classify button with popup launcher
# classify_btn = tk.Button(articles_tab, text="Classify all articles", command=model_selection_popup)
classify_btn = tk.Button(articles_tab, text="LLM Classify all articles", command=lambda: classif_model_selection_popup(root))
classify_btn.pack(pady=5)

# subjects dropdown to select by the user for the dataset generation
tk.Label(articles_tab, text="Select Subject for Generation:").pack()
selected_subject_var = tk.StringVar()
subject_dropdown = ttk.Combobox(articles_tab, textvariable=selected_subject_var, state="readonly", width=40)
subject_dropdown.pack()

# Button to train models is defined after the function for training models...

# --- Frame for Treeview + Scrollbar ---
tree_frame = tk.Frame(articles_tab)
tree_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Scrollbar (define first, pack later)
scrollbar = tk.Scrollbar(tree_frame, orient="vertical")

# Treeview
articles_tree = ttk.Treeview(
    tree_frame,
    columns=("id", "title", "subject", "label", "polarity", "subjectivity", "v_compound", "sentiment", "source"),
    show="headings",
    height=20,
    yscrollcommand=scrollbar.set
)
scrollbar.config(command=articles_tree.yview)
# Pack order is important!
articles_tree.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Bind double-click
articles_tree.bind("<Double-1>", on_article_double_click) # on double-click copy current article title to search field

# Scrollbar (place after tree so it's next to it)
# scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=articles_tree.yview)
# scrollbar.pack(side="right", fill="y")
# articles_tree.config(yscrollcommand=scrollbar.set)


# Define column widths and headings
column_widths = {
    "no": 40,
    "title": 420,
    "subject": 120,
    "label": 60,
    "polarity": 80,
    "subjectivity": 80,
    "v_compound": 80,
    "sentiment": 80,
    #"confidence": 80,
    #"sentiment": 80,
    #"score": 60,
    "source": 100
}
for col in articles_tree["columns"]:
    # articles_tree.heading(col, text=col.title()) # following line is to sort the grid..
    articles_tree.heading(col, text=col.title(), command=lambda c=col: sort_treeview_column(c, False))

    articles_tree.column(col, width=column_widths.get(col, 100), anchor="center")



# === Users Tab ===
users_tab = ttk.Frame(notebook)
notebook.add(users_tab, text="Users")

# Form Inputs
tk.Label(users_tab, text="Name").grid(row=0, column=0, padx=5, pady=5)
tk.Label(users_tab, text="Email").grid(row=0, column=1, padx=5, pady=5)
tk.Label(users_tab, text="Role").grid(row=0, column=2, padx=5, pady=5)

user_name_entry = tk.Entry(users_tab, width=25)
user_email_entry = tk.Entry(users_tab, width=25)
user_role_combobox = ttk.Combobox(users_tab, values=["user", "moderator", "admin"], state="readonly", width=15)
user_role_combobox.current(0)

user_name_entry.grid(row=1, column=0, padx=5)
user_email_entry.grid(row=1, column=1, padx=5)
user_role_combobox.grid(row=1, column=2, padx=5)

tk.Button(users_tab, text="Add User", command=lambda: add_user()).grid(row=1, column=3, padx=5)
tk.Button(users_tab, text="Delete Selected User", command=lambda: delete_user()).grid(row=1, column=4, padx=5)
tk.Button(users_tab, text="Refresh", command=lambda: refresh_users()).grid(row=1, column=5, padx=5)

users_tree = ttk.Treeview(
    users_tab,
    columns=("id", "name", "email", "role", "joined"),
    show="headings",
    height=20
)
for col in users_tree["columns"]:
    users_tree.heading(col, text=col.title())
    users_tree.column(col, anchor="center", width=120)
users_tree.grid(row=2, column=0, columnspan=6, padx=10, pady=10)



# === Subjects Tab ===
subjects_tab = ttk.Frame(notebook)
notebook.add(subjects_tab, text="Subjects")

tk.Label(subjects_tab, text="New Subject:").grid(row=0, column=0, padx=5, pady=5)
subject_entry = tk.Entry(subjects_tab, width=30)
subject_entry.grid(row=0, column=1, padx=5, pady=5)

subject_add_button = tk.Button(subjects_tab, text="Add Subject", command=lambda: add_subject())
subject_add_button.grid(row=0, column=2, padx=5, pady=5)
subject_delete_button = tk.Button(subjects_tab, text="Delete Selected", command=lambda: delete_subject())
subject_delete_button.grid(row=0, column=3, padx=5, pady=5)
subject_refresh_button = tk.Button(subjects_tab, text="Refresh Subjects", command=lambda: refresh_subjects())
subject_refresh_button.grid(row=0, column=4, padx=5, pady=5)

subjects_tree = ttk.Treeview(
    subjects_tab,
    columns=("subject",),
    show="headings",
    height=20
)
subjects_tree.heading("subject", text="Subject")
subjects_tree.column("subject", anchor="center", width=200)
subjects_tree.grid(row=1, column=0, columnspan=5, padx=10, pady=10)


# === Models Tab ===
models_tab = ttk.Frame(notebook)
notebook.add(models_tab, text="Models")

# Frame to hold the models tree and its related buttons in the Models tab
models_tab_frame = tk.Frame(models_tab)
models_tab_frame.pack(fill="both", expand=True, padx=10, pady=10)

models_tree = ttk.Treeview(
    # models_tab,
    models_tab_frame,
    columns=("model_name", "timestamp", "use_embeddings", "use_tfidf"),
    show="headings",
    # height=20
)
models_tree.heading("model_name", text="Model Name")
models_tree.heading("timestamp", text="Date")
models_tree.heading("use_embeddings", text="Embeddings")
models_tree.heading("use_tfidf", text="TF-IDF")

for col in ("model_name", "timestamp", "use_embeddings", "use_tfidf"):
    models_tree.column(col, anchor="center", width=150)

# models_tree.pack(fill="both", expand=True, padx=10, pady=10)
models_tree.pack(fill="both", expand=True)
# Delete model button just below the table
model_delete_button = tk.Button(models_tab, text="Delete Selected Model", command=lambda: delete_model())
model_delete_button.pack(pady=5)

# === Additional Functions ===

def refresh_subject_dropdown():
    subject_list = [s["name"] for s in subjects_col.find()]
    subject_dropdown["values"] = subject_list
    if subject_list:
        selected_subject_var.set(subject_list[0])


Kaggle_data_dir = "./fakenews_datasets/Kaggle"
Kaggle_dataset_name = "AllNews_Kaggle.csv"
Kaggle_dataset_path = os.path.join(Kaggle_data_dir, Kaggle_dataset_name)

from itertools import islice
from dateutil import parser

# Upload Kaggle fakenews articles dataset (or a part of it)
def import_kaggle_articles1():
    MaxArticles = 45000
    if not os.path.exists(Kaggle_dataset_path):
        messagebox.showerror("File Not Found", f"CSV not found at: {Kaggle_dataset_path}")
        return

    inserted_count = 0
    batch = []
    batch_size = 200

    with open(Kaggle_dataset_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(tqdm(islice(reader, MaxArticles), desc="Importing Kaggle Articles")):
            if i >= MaxArticles:
                break
            try:
                label = int(row["label"])
                try:
                    article_date = parser.parse(row["date"])
                except Exception:
                    article_date = datetime.now()

                article = {
                    "title": row["title"],
                    "text": row["text"],
                    "subject": row["subject"],
                    "date": article_date,
                    "label": label,
                    "polarity": 0,
                    "subjectivity": 0,
                    "v_compound": 0,
                    "sentiment": "neutral",
                    "source": "Kaggle",
                    "submitted_by": None,
                    "reviewed": False,
                    "comments": []
                }

                batch.append(article)

                if len(batch) >= batch_size:
                    articles_col.insert_many(batch)
                    inserted_count += len(batch)
                    batch.clear()

            except Exception as e:
                print(f"❌ Error in row {i}: {e}")
                print(f"   Problematic data → title: {row.get('title', '')[:50]} | date: {row.get('date', '')}")
                continue

    # insert remaining articles
    if batch:
        articles_col.insert_many(batch)
        inserted_count += len(batch)

    messagebox.showinfo("Import Complete", f"Imported {inserted_count} articles from Kaggle.")
    #
    log_user_action("import_dataset", f"Imported synthetic dataset from {Kaggle_dataset_path}")




synthetic_data_dir = "./fakenews_datasets/Synthetic"  # "./fakenews_datasets"
synthetic_dataset_name = "Synthetic_FakeNews_Dataset.csv"   # "FakeNewsNet.csv"
synthetic_path = os.path.join(synthetic_data_dir, synthetic_dataset_name)
api_key = os.environ.get("NEWSAPI_KEY")

# === Synthetic Dataset Generation Function ===
def generate_synthetic_dataset(api_key, selected_subject):
    if not api_key:
        messagebox.showerror("API key required", "Set NEWSAPI_KEY before launching the app.")
        return

    os.makedirs(synthetic_data_dir, exist_ok=True)

    query = selected_subject if selected_subject else "technology"
    # Use subject filter from subject dropdown or default
    subject_list = [s["name"] for s in subjects_col.find()]
    Selected_subjects = " OR ".join(subject_list) if subject_list else "technology"

    df = create_semi_synthetic_news_dataset(
        api_key=api_key,
        query=query,
        save_path=synthetic_path,
        n_per_class=95
    )
    if df is not None:
        messagebox.showinfo("Done", f"Synthetic dataset created with {len(df)} articles.")
        log_user_action("import_dataset", f"generate synthetic dataset {synthetic_path}")

    else:
        messagebox.showerror("Error", "Failed to generate synthetic dataset.")



# === Model Training Option Popup ===

# create a dataframe from the filtered grid
'''
    # without any call to MongoDB...
    def get_filtered_articles_from_treeview():
    articles = []
    for child in articles_tree.get_children():
        values = articles_tree.item(child)["values"]
        # Example: [id, title, text, label]
        article = {
            "title": values[1],
            "text": values[2],
            "label": values[3]
        }
        articles.append(article)
    return pd.DataFrame(articles)

'''
def get_filtered_articles_from_treeview():
    titles = [articles_tree.item(child)["values"][1] for child in articles_tree.get_children()]
    if not titles:
        print("⚠ No titles selected in Treeview.")
        return pd.DataFrame()
    # Filters only matching titles ($in: titles), gets back specific fields from articles collection
    articles = list(articles_col.find(
    {"title": {"$in": titles}},
    {"title": 1, "text": 1, "label": 1, "subject": 1, "_id": 0}
))
    return pd.DataFrame(articles)

# model training analysis block
def open_model_training_options():

    def on_train():
        use_embed = var.get() == 1

        # optional indexing on title that speeds up the calls filtered by title to MongoDB =...
        # articles_col.create_index("title")  SEE THIS !!!

        # create a dataframe from the treeview grid making also a call to MongoDB
        df = get_filtered_articles_from_treeview()
        print(f"🔍 Retrieved {len(df)} filtered articles from MongoDB.")

        if df.empty:
            messagebox.showwarning("No articles", "Import or select labeled articles before training.")
            return
        try:
            Train_Models_from_dataframe(df, filtered_option=True, use_embeddings=use_embed)
        except ValueError as error:
            messagebox.showerror("Training could not start", str(error))
            return
        popup.destroy()
        messagebox.showinfo("Training Complete", f"Models trained using {'embeddings' if use_embed else 'TF-IDF'}.")
        #
        log_user_action("train_models", f"Trained on {len(df)} filtered articles using {'embeddings' if use_embed else 'TF-IDF'}")


    popup = tk.Toplevel(root)
    popup.title("Model Training Options")
    popup.geometry("300x200")

    var = tk.IntVar()
    var.set(0)  # default to TF-IDF

    tk.Label(popup, text="Choose feature extraction method:", font=("Arial", 12)).pack(pady=10)
    tk.Radiobutton(popup, text="TF-IDF (default)", variable=var, value=0).pack(anchor="w", padx=20)
    tk.Radiobutton(popup, text="Embeddings (SentenceTransformers)", variable=var, value=1).pack(anchor="w", padx=20)

    tk.Button(popup, text="Train Models", command=on_train).pack(pady=20)



# Sentiment analysis on articles filtered or not...
def run_sentiment_analysis_on_mongodb_articles():
    # see next remarks if use filtered articles...
    # df = get_filtered_articles_from_treeview()
    # sentiment_analysis_on_articles(change df=None to df=df, ...)
    sentiment_analysis_on_articles(df=None, save_in_mongoDB=True,
                                   save_csv_path="./fakenews_datasets/sentiment/articles_sentiments.csv")
    refresh_articles()

    messagebox.showinfo("Done", "Sentiment analysis completed and stored in MongoDB.")
    #
    log_user_action("sentiment_analysis", "Sentiment analysis run on MongoDB articles")

def create_admin_buttons():
    admin_button_frame = tk.Frame(articles_tab)
    admin_button_frame.pack(pady=5)
    tk.Button(admin_button_frame, text="Generate Synthetic Dataset", command=lambda: generate_synthetic_dataset(api_key, selected_subject_var.get())).pack(side="left", padx=5)
    tk.Button(admin_button_frame, text="Upload Synthetic/other dataset", command=lambda: run_csv_import()).pack(side="left", padx=5)


models_frame = tk.Frame(articles_tab)
models_frame.pack(pady=5)

tk.Button(models_frame, text="Train Models from MongoDB Articles", command=open_model_training_options).pack(side="left", padx=10)
tk.Button(models_frame, text="Run Sentiment Analysis", command=run_sentiment_analysis_on_mongodb_articles).pack(side="left", padx=10)



# --- Model Selection Popup with Callback ---
def classif1_model_selection_popup(root, on_confirm_callback):
    popup = tk.Toplevel(root)
    popup.title("Select LLM Model")
    popup.geometry("300x250")

    tk.Label(popup, text="Choose a model for classification:", font=("Arial", 10, "bold")).pack(pady=10)

    model_var = tk.StringVar(value="bert")
    use_full_text_var = tk.BooleanVar(value=False)

    models = [
        ("BERT (mrm8488)", "bert"),
        ("FLAN-T5", "flan"),
        ("BART-base", "bart"),
        ("RoBERTa (hamzab)", "roberta")
    ]
    for label, value in models:
        tk.Radiobutton(popup, text=label, variable=model_var, value=value).pack(anchor="w", padx=20)

    tk.Checkbutton(popup, text="Use full text (title + body)", variable=use_full_text_var).pack(pady=10)

    def on_confirm():
        model_type = model_var.get()
        use_full_text = use_full_text_var.get()
        popup.destroy()
        on_confirm_callback(model_type, use_full_text)

    tk.Button(popup, text="Instant Classify Article", command=on_confirm).pack(pady=10)

'''
# classify one article found in search_entry field with the user selected model in the above function
def classify_article_llm():
    title = search_entry.get().strip()
    if not title:
        messagebox.showwarning("Input in search entry Required", "Please enter an article title to classify.")
        return
    try:
        classif1_model_selection_popup(root)

        prediction = classify_with_mrm8488_bert(title)
        icon = "✅ Real" if prediction["label"].lower() == "real" else "❌ Fake"
        llm_result_var.set(f"{icon} {prediction['label']} ({prediction['confidence']:.2f})")
    except Exception as e:
        messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
        llm_result_var.set("")  # Clear result if error
'''
def classify_article_llm():
    title = search_entry.get().strip()
    if not title:
        messagebox.showwarning("Input Required", "Please enter an article title to classify.")
        return

    def handle_classification(model_type, use_full_text):
        try:
            # You could use title + body here if needed and available
            if model_type == "bert":
                prediction = classify_with_mrm8488_bert(title)
            elif model_type == "flan":
                prediction = classify_with_llm_flan_t5(title)
            elif model_type == "bart":
                prediction = classify_with_bart(title)
            elif model_type == "roberta":
                prediction = classify_with_roberta(title)
            else:
                raise ValueError("Unknown model type selected")

            icon = "✅ Real" if prediction["label"].lower() == "real" else "❌ Fake"
            llm_result_var.set(f"{icon} {prediction['label']} ({prediction['confidence']:.2f})")
            #
            log_user_action("classify_article", f"Classified article '{title[:50]}' using model '{model_type}'")


        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
            llm_result_var.set("")

    # Open the popup and pass the handler
    classif1_model_selection_popup(root, handle_classification)


# === Functions managing MongoDB tables ===

def refresh_articles():
     # Save the yview (vertical scroll position)
    y_pos = articles_tree.yview()
    # clear tree...
    articles_tree.delete(*articles_tree.get_children())
    # Refill tree with articles
    for i, art in enumerate(articles_col.find(), start=1):
        articles_tree.insert("", "end", values=(
            i,  # Row number instead of ObjectId
            art.get("title", "")[:200],
            art.get("subject", ""),
            "✅ Real" if art.get("label") == 1 else "❌ Fake",
            round(art.get("polarity", 0.0), 2),
            round(art.get("subjectivity", 0.0), 2),
            round(art.get("v_compound", 0.0), 2),
            art.get("sentiment", ""),   #  is it correct ??? maybe this ?  art.get("sentiment", "neutral"),
            art.get("source", "")
        ))
    # Restore scroll position
    articles_tree.yview_moveto(y_pos[0])


def search_articles():
    field = search_field_var.get()
    query = search_entry.get().strip()
    if not query:
        refresh_articles()
        return
    condition = {field: {"$regex": query, "$options": "i"}} if field not in ["polarity", "subjectivity", "v_compound"] else {
        field: {"$gte": float(query)}} # Numeric comparison (you can refine it)
    results = articles_col.find(condition)

    articles_tree.delete(*articles_tree.get_children())
    for i, art in enumerate(results, start=1):
        articles_tree.insert("", "end", values=(
            i,
            art.get("title", "")[:200],
            art.get("subject", ""),
            "✅ Real" if art.get("label") == 1 else "❌ Fake",
            round(art.get("polarity", 0.0), 2),
            round(art.get("subjectivity", 0.0), 2),
            round(art.get("v_compound", 0.0), 2),
            art.get("sentiment", ""), # also chatgpt suggests : art.get("sentiment", "neutral")
            art.get("source", "")
        ))
    log_user_action("search", f"Search for '{query}' in field '{field}'")


# from bson import ObjectId  <--- used only for _id matching
# delete selected articles : the PROBLEM here is the matching of titles from tree+view to the original mongodb articles
# Matching by title is not 100% safe if titles are not unique — but if they are in your case, it’s acceptable.
def delete_selected_articles():
    selected_items = articles_tree.selection()
    if not selected_items:
        print("⚠ No articles selected for deletion.")
        return

    deleted_count = 0

    for item in selected_items:
        values = articles_tree.item(item)["values"]
        title = values[1]  # Title assumed to be second column

        try:
            result = articles_col.delete_one({"title": title})
            if result.deleted_count == 1:
                articles_tree.delete(item)
                deleted_count += 1
            else:
                print(f"⚠ No article found in DB with title: {title}")
        except Exception as e:
            print(f"❌ Error deleting article with title '{title}': {e}")

    print(f"🗑️ Deleted {deleted_count} article(s) by title.")
    if current_user:
        log_user_action("delete_article", f"Deleted {deleted_count} article(s) by title from MongoDB.")



# sort treeview by a specific column
def sort_treeview_column(col, reverse):
    items = [(articles_tree.set(k, col), k) for k in articles_tree.get_children("")]
    try:
        items.sort(key=lambda t: float(t[0]), reverse=reverse)
    except ValueError:
        items.sort(reverse=reverse)
    for index, (val, k) in enumerate(items):
        articles_tree.move(k, "", index)
    articles_tree.heading(col, command=lambda: sort_treeview_column(col, not reverse))


# button to run search a field in articles grid...
tk.Button(search_frame, text="Search", command=search_articles).pack(side="left", padx=5)
tk.Button(search_frame, text="Refresh Articles", command=refresh_articles).pack(side="left", padx=5)
# when button selected the classify article in search_field...
tk.Button(search_frame, text="LLM Classify", command=classify_article_llm).pack(side="left", padx=5)

tk.Button(search_frame, text="🗑 Delete Selected Articles", command=lambda: delete_selected_articles()).pack(side="left", padx=5)


# empty database' collections and create a few records for users, the required subjects and one article.
# after that add all Kaggle articles in the articles table...
def setup_sample_data():
    if not messagebox.askyesno(
        "Reset demo database",
        "This clears articles, users, subjects, models and logs in the local demo database. Continue?",
    ):
        return

    db.articles.drop()
    db.users.drop()
    db.comments.drop()
    db.subjects.drop()
    db.models.drop()
    db.user_logs.drop()

    subjects = ["Politics", "Health", "Technology", "Economy", "Environment", "Culture"]
    db.subjects.insert_many([{"name": s} for s in subjects])

    users = [
        {"_id": uuid.uuid4().hex, "name": "Alice Moderator", "email": "alice@newsapp.com",    "role": "moderator",   "password": "alice123",  "joined": datetime.now()},
        {"_id": uuid.uuid4().hex, "name": "Bob Admin",       "email": "admin@example.test",      "role": "admin",       "password": "local-demo",       "joined": datetime.now()},
        {"_id": uuid.uuid4().hex, "name": "Charlie User",    "email": "charlie@newsapp.com",  "role": "user",        "password": "charlie",   "joined": datetime.now()},
    ]
    db.users.insert_many(users)
    # models = {"model_name": "fake (init) model", "timestamp": datetime.now(), "use_embeddings": "✅", "use_tfidf": "❌"}
    models = {"model_name": "fake (init) model 1", "timestamp": datetime.now(), "use_embeddings": True,  "use_tfidf": False}
    db.models.insert_one(models)

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

    import_kaggle_articles1()
    refresh_articles()
    refresh_users()
    refresh_subjects()
    refresh_models()
    messagebox.showinfo("Success", "Sample data has been reset and loaded.")


def refresh_users():
    users_tree.delete(*users_tree.get_children())
    for user in users_col.find():
        users_tree.insert("", "end", values=(
            str(user.get("_id", ""))[:6],
            user.get("name", ""),
            user.get("email", ""),
            user.get("role", ""),
            user.get("joined", "").strftime("%Y-%m-%d") if user.get("joined") else ""
        ))

def add_user():
    name = user_name_entry.get().strip()
    email = user_email_entry.get().strip()
    role = user_role_combobox.get()

    if not name or not email:
        messagebox.showwarning("Input Error", "Please enter both name and email.")
        return

    users_col.insert_one({
        "name": name,
        "email": email,
        "role": role,
        "joined": datetime.now()
    })

    user_name_entry.delete(0, tk.END)
    user_email_entry.delete(0, tk.END)
    refresh_users()

def delete_user():
    selected = users_tree.selection()
    if not selected:
        messagebox.showwarning("No Selection", "Select a user to delete.")
        return

    user_id = users_tree.item(selected[0])["values"][0]
    users_col.delete_one({"_id": {"$regex": f"^{user_id}"}})
    refresh_users()

def refresh_subjects():
    subjects_tree.delete(*subjects_tree.get_children())
    for subj in subjects_col.find():
        subjects_tree.insert("", "end", values=(subj.get("name", ""),))

def add_subject():
    name = subject_entry.get().strip()
    if not name:
        messagebox.showwarning("Input Error", "Please enter a subject name.")
        return
    if subjects_col.find_one({"name": name}):
        messagebox.showinfo("Exists", "This subject already exists.")
        return
    subjects_col.insert_one({"name": name})
    subject_entry.delete(0, tk.END)
    refresh_subjects()
    refresh_subject_dropdown()  # update dropdown too

def delete_subject():
    selected = subjects_tree.selection()
    if not selected:
        messagebox.showwarning("No Selection", "Select a subject to delete.")
        return
    subject_name = subjects_tree.item(selected[0])["values"][0]
    subjects_col.delete_one({"name": subject_name})
    refresh_subjects()


def refresh_models():
    models_tree.delete(*models_tree.get_children())
    models_col = db["models"]
    for m in models_col.find().sort("timestamp", -1):
        models_tree.insert("", "end", values=(
            os.path.basename(m.get("model_name", "")),
            m.get("timestamp", "").strftime("%Y-%m-%d %H:%M") if m.get("timestamp") else "",
            "✅" if m.get("use_embeddings", False) else "❌",
            "✅" if m.get("use_tfidf", False) else "❌"

        ))


def delete_model():
    selected_item = models_tree.selection()
    if not selected_item:
        messagebox.showwarning("No Selection", "Please select a model to delete.")
        return
    confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the selected model?")
    if not confirm:
        return
    item = models_tree.item(selected_item)
    model_name = item["values"][0]
    if not model_name:
        messagebox.showerror("Error", "Could not retrieve model name.")
        return
    # result = db["models"].delete_one({"model_name": {"$regex": model_name}}) ??
    # result = db["models"].delete_one({"model_name": {"$regex": f"^{model_name}$", "$options": "i"}}) case insensitive "f"=="F"
    result = db["models"].delete_one({"model_name": model_name})   # exact match
    if result.deleted_count:
        messagebox.showinfo("Deleted", f"Model '{model_name}' deleted successfully.")
    else:
        messagebox.showwarning("Not Found", f"Model '{model_name}' not found in database.")
    refresh_models()


# --- Load Initial Data ---
refresh_articles()
refresh_users()
refresh_subjects()
refresh_subject_dropdown()
refresh_models()

root.mainloop()
