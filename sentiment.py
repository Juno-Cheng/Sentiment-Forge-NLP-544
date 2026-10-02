"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
This is an unimplemented student starter, not a solution or an autograder.
"""
import csv
import math
from collections import Counter
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import html
import contractions

#=============== Custom Functions ===============
#--- prepare_data helpers (Section 1) ---
def prepare_data_read_data(path): # read the data from the file and return a pandas dataframe
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, on_bad_lines="skip", usecols=["review_body", "star_rating"])
    df = df.rename(columns={"review_body": "review", "star_rating": "rating"})
    return df

def prepare_data_convert_and_remove_invalid_rows(df):
    #convert the rating to a numeric value
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # filter out the rows where the review is empty or the rating is not between 1 and 5
    valid_rows = (df["review"] != "") & df["rating"].isin([1, 2, 3, 4, 5])

    # count the number of rows that were excluded
    excluded_rows = int((~valid_rows).sum())

    # filter out the rows where the review is empty or the rating is not between 1 and 5
    df = df.loc[valid_rows].copy()

    # convert the rating to an integer
    df["rating"] = df["rating"].astype(int)
    return df, excluded_rows

def prepare_data_count_ratings_for_stats(df):
    # count the number of rows for each rating
    rating_counts = {i: int((df["rating"] == i).sum()) for i in range(1, 6)}
    class_counts = {
        "positive": int((df["rating"] >= 4).sum()),
        "negative": int((df["rating"] <= 2).sum()),
        "neutral": int((df["rating"] == 3).sum()),
    }
    return rating_counts, class_counts

def prepare_data_train_test_split(df, n_per_class):
    # Sample the data for each rating ignoring the neutral class (this will make it evenly distributed)
    pos_df = df.loc[df["rating"] >= 4].sample(
        n=n_per_class, replace=False, random_state=42
    )
    neg_df = df.loc[df["rating"] <= 2].sample(
        n=n_per_class, replace=False, random_state=42
    )
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df["label"] = 1
    neg_df["label"] = 0

    # Concatenate the positive and negative dataframes and add a label column
    sampled = pd.concat([pos_df, neg_df], ignore_index=True)[
        ["review", "rating", "label"]
    ]

    # Split the data into train and test sets
    train_df, test_df = train_test_split(
        sampled,
        test_size=0.2,
        stratify=sampled["label"],
        random_state=42,
    )
    return train_df, test_df

#--- clean_review helpers (Section 2) ---
def clean_review_unescape_html(text):
    text = html.unescape(text) # Given by the handout
    text = text.lower()
    return text

def clean_review_replace_html_and_url(text):
    text = re.sub(r"<[^>]*>", " ", text)  # Given by the handout
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Given by the handout
    return text

def clean_review_replace_curly_apostrophes(text):
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # ‘ and ’
    return text

def clean_review_expand_contractions(text):
    text = contractions.fix(text)
    return text

def clean_review_keep_only_a_z_and_spaces_collapse_whitespace_runs_strip_ends(text):
    text = re.sub(r"[^a-z\s]", " ", text)  # Given by the handout
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

#--- preprocess_review helpers (Section 3) ---
def preprocess_review_split_and_remove_stopwords(text):
    # (a) split on whitespace; drop NLTK English stopwords except no, nor, not
    tokens = text.split()
    stop = set(stopwords.words("english")) - {"no", "nor", "not"}
    tokens = [token for token in tokens if token not in stop]
    return tokens

def preprocess_review_lemmatize(tokens):
    # (b) WordNetLemmatizer.lemmatize each token (default noun); no POS tagging
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def preprocess_review_join(tokens):
    # (c) join tokens with single spaces
    text = " ".join(tokens)
    return text

#--- extract_features helpers (Section 4) ---
def extract_features_default(train_texts, test_texts):
    # defaults = unigrams, sublinear_tf=False, smooth IDF, L2
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)  # learn vocab/IDF on train only
    X_test = vectorizer.transform(test_texts)        # unseen/empty → zero row
    return vectorizer, X_train, X_test

#--- train_model helpers (Section 5) ---
def train_model_perceptron(X_train, y_train):
    model = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    return model

def train_model_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000, tol=1e-4)
    model.fit(X_train, y_train)
    return model

def train_model_naive_bayes(X_train, y_train):
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model

#=============== HELPER FUNCTIONS ===============
def prepare_data(path, n_per_class=100000):
    """Return (train_df, test_df, stats), as specified in Section 1/interface."""

    # 1.  read the data from the file
    df = prepare_data_read_data(path)

    # 2. convert the rating to a numeric value and remove the invalid rows
    df, excluded_rows = prepare_data_convert_and_remove_invalid_rows(df)

    # 3. count the number of rows for each rating for stats
    rating_counts, class_counts = prepare_data_count_ratings_for_stats(df)

    # Stats Creation
    stats = {
        "rating_counts": rating_counts,
        "class_counts": class_counts,
        "excluded_rows": excluded_rows,
    }

    # 4. train and test split
    train_df, test_df = prepare_data_train_test_split(df, n_per_class)

    return train_df, test_df, stats

def clean_review(text):
    """Return one cleaned string, including the empty string when applicable."""
    # 1. unescape HTML entities (e.g. &#34; → ") then lowercase everything
    text = clean_review_unescape_html(text)

    # 2. replace HTML tags <...>→ one space replace URLs http(s)://... or www.... → one space
    text = clean_review_replace_html_and_url(text)
    
    # 3. curly apostrophes ’ ‘      → straight '
    text = clean_review_replace_curly_apostrophes(text)

    # 4. expand contractions (won't → will not) via contractions.fix
    text = clean_review_expand_contractions(text)

    # 5. keep only a–z and spaces (drop digits, punctuation, etc.) collapse whitespace runs → single spaces strip ends
    text = clean_review_keep_only_a_z_and_spaces_collapse_whitespace_runs_strip_ends(text)
    
    return text

def preprocess_review(text):
    """Return stopword-filtered, noun-lemmatized text from cleaned input."""
    # 1. split on whitespace; remove stopwords (the,a, um) but (keep no, nor, not)
    tokens = preprocess_review_split_and_remove_stopwords(text)

    # 2. lemmatize remaining tokens (WordNet, default noun)
    # Ex. running → run
    tokens = preprocess_review_lemmatize(tokens)

    # 3. join with single spaces and return
    text = preprocess_review_join(tokens)

    return text

def extract_features(train_texts, test_texts):
    """Return (fitted_vectorizer, sparse_X_train, sparse_X_test)."""
    # defaults = unigrams, sublinear_tf=False, smooth IDF, L2
    vectorizer, X_train, X_test = extract_features_default(train_texts, test_texts)
    return vectorizer, X_train, X_test

def train_model(name, X_train, y_train):
    """Return a fitted classifier for one of the three specified model names."""
    # "Perceptron"         -> Perceptron(random_state=42, max_iter=1000, tol=1e-3)
    # "Logistic Regression"-> LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000, tol=1e-4)
    # "Naive Bayes"        -> MultinomialNB(alpha=1.0)   -> train_model_naive_bayes(X_train, y_train)
    if name == "Perceptron":
        return train_model_perceptron(X_train, y_train)
    elif name == "Logistic Regression":
        return train_model_logistic_regression(X_train, y_train)
    elif name == "Naive Bayes":
        return train_model_naive_bayes(X_train, y_train)
    else:
        raise ValueError(f"Invalid model name: {name}")

def evaluate_model(model, X, y):
    """Return unrounded accuracy, precision, recall, f1 in a dictionary."""
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(
            y, preds, average="binary", pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y, preds, average="binary", pos_label=1, zero_division=0
        ),
        "f1": f1_score(y, preds, average="binary", pos_label=1, zero_division=0),
    }


def fit_bigram(train_texts, min_count=3):
    """Return a dictionary with vocab, bigrams, contexts; omit zero counts."""
    # 1. tokenize each review (whitespace); build vocab (count >= min_count)
    tokenized = [text.split() for text in train_texts]
    word_counts = Counter()
    for tokens in tokenized:
        word_counts.update(tokens)
    retained = {w for w, c in word_counts.items() if c >= min_count}

    # 2. V = retained words + <unk> + </s>  (<s> is context only, not in V)
    vocab = retained | {"<unk>", "</s>"}

    # 3. map OOV -> <unk>; wrap each seq with <s> ... </s>
    # 4. count bigrams C(h,w) and contexts C(h); omit zeros
    bigrams = Counter()
    contexts = Counter()
    # iterate over each tokenized text
    for tokens in tokenized:
        mapped = [w if w in retained else "<unk>" for w in tokens] # map OOV -> <unk>
        seq = ["<s>"] + mapped + ["</s>"] # wrap each seq with <s> ... </s>

        # count bigrams C(h,w) and contexts C(h); omit zeros
        for i in range(len(seq) - 1):
            h, w = seq[i], seq[i + 1]
            bigrams[(h, w)] += 1
            contexts[h] += 1

    return {"vocab": vocab, "bigrams": bigrams, "contexts": contexts}


def evaluate_bigram(model, test_texts, k=1.0):
    """Return a dictionary with unseen, n_tokens, add-k perplexity; do not refit."""
    vocab = model["vocab"]
    bigrams = model["bigrams"]
    contexts = model["contexts"]
    retained = vocab - {"<unk>", "</s>"}
    v_size = len(vocab)

    # 1. map test with model vocab (<unk>, <s>, </s>); do not update counts
    unseen = 0
    n_tokens = 0
    log_sum = 0.0
    for text in test_texts:
        tokens = text.split()
        mapped = [w if w in retained else "<unk>" for w in tokens]
        seq = ["<s>"] + mapped + ["</s>"]
        # 2–3. score each transition with add-k; count unseen / N
        for i in range(len(seq) - 1):
            h, w = seq[i], seq[i + 1]
            c_hw = bigrams.get((h, w), 0)
            c_h = contexts.get(h, 0)
            if c_hw == 0:
                unseen += 1
            p = (c_hw + k) / (c_h + k * v_size)
            log_sum += math.log(p)
            n_tokens += 1

    perplexity = math.exp(-log_sum / n_tokens)
    return {"unseen": unseen, "n_tokens": n_tokens, "perplexity": perplexity}

#=============== Unit Test Code  ===============

#=============== MAIN FUNCTION ===============
def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""

    # 1. Dataset Preparation

    # 2. Data Cleaning

    # 3. Preprocessing

    # 4. Feature Extraction

    # 5. Models (Perceptron, Logistic Regression, Naive Bayes)

    # 6. Bigram Model & Smoothing (Add-k)

    raise NotImplementedError


if __name__ == "__main__":
    run_experiment()
