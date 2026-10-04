"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
This is an unimplemented student starter, not a solution or an autograder.
"""
import csv
import html
import math
import re
from collections import Counter

import contractions
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#=============== Custom Functions ===============
#--- prepare_data helpers (Section 1) ---
def prepare_data_read_data(path):  # read TSV; keep only review + rating columns
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        on_bad_lines="skip",
        usecols=["review_body", "star_rating"],
    )
    df = df.rename(columns={"review_body": "review", "star_rating": "rating"})
    return df

def prepare_data_convert_and_remove_invalid_rows(df):
    # convert the rating to a numeric value
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # filter out the rows where the review is empty or the rating is not between 1 and 5
    valid_rows = (df["review"] != "") & df["rating"].isin([1, 2, 3, 4, 5])

    # count the number of rows that were excluded
    excluded_rows = int((~valid_rows).sum())

    # keep only valid rows
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

    # Split the data into train and test sets (80% train, 20% test - both with negative and positive labels)
    train_df, test_df = train_test_split(
        sampled,
        test_size=0.2,
        stratify=sampled["label"],
        random_state=42,
    )
    return train_df, test_df

#--- clean_review helpers (Section 2) ---
def clean_review_unescape_html(text):
    text = html.unescape(text)  # given by the handout
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
_STOPWORDS = set(stopwords.words("english")) - {"no", "nor", "not"}
_LEMMATIZER = WordNetLemmatizer()

def preprocess_review_split_and_remove_stopwords(text):
    # (a) split on whitespace; drop NLTK English stopwords except no, nor, not
    tokens = text.split()
    tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens

def preprocess_review_lemmatize(tokens):
    # (b) WordNetLemmatizer.lemmatize each token (default noun); no POS tagging
    tokens = [_LEMMATIZER.lemmatize(token) for token in tokens]
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

#--- print functions (required 37 lines) ---
def print_data_and_features(stats, avg_before_clean, avg_after_clean,
                            avg_before_pre, avg_after_pre, tfidf_shape):
    print(f"Positive reviews: {stats['class_counts']['positive']}")
    print(f"Negative reviews: {stats['class_counts']['negative']}")
    print(f"Neutral reviews: {stats['class_counts']['neutral']}")
    print(f"Average length before cleaning: {avg_before_clean:.4f}")
    print(f"Average length after cleaning: {avg_after_clean:.4f}")
    print(f"Average length before preprocessing: {avg_before_pre:.4f}")
    print(f"Average length after preprocessing: {avg_after_pre:.4f}")
    print(f"TF-IDF matrix shape: ({tfidf_shape[0]}, {tfidf_shape[1]})")

def print_model_performance(name, train_metrics, test_metrics):
    print(f"{name} Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"{name} Training Precision: {train_metrics['precision']:.4f}")
    print(f"{name} Training Recall: {train_metrics['recall']:.4f}")
    print(f"{name} Training F1-score: {train_metrics['f1']:.4f}")
    print(f"{name} Testing Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"{name} Testing Precision: {test_metrics['precision']:.4f}")
    print(f"{name} Testing Recall: {test_metrics['recall']:.4f}")
    print(f"{name} Testing F1-score: {test_metrics['f1']:.4f}")

def print_bigram_model_and_smoothing(bigram_model, eval_k1, eval_k01, eval_k001):
    # vocab from fit_bigram; unseen/perplexity from evaluate_bigram
    print(f"Bigram vocabulary size: {len(bigram_model['vocab'])}")
    print(f"Bigram unseen test transitions: {eval_k1['unseen']}")
    print(f"Bigram test perplexity: {eval_k1['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.1): {eval_k01['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.01): {eval_k001['perplexity']:.4f}")

#=============== STEP BY STEP FUNCTIONS ===============
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
    # tokenized: list of reviews; each review is a list of words, e.g. ["the", "pen", "is", "good"]
    tokenized = []
    for text in train_texts:
        tokenized.append(text.split())
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

#=============== MAIN FUNCTION ===============
def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""

    # 1. Dataset Preparation
    train_df, test_df, stats = prepare_data(path, n_per_class)

    # 2. Data Cleaning - Remove HTML tags, URLs, and other unwanted characters
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["cleaned"] = train_df["review"].apply(clean_review)
    test_df["cleaned"] = test_df["review"].apply(clean_review)

    # 3. Preprocessing - Remove stopwords and lemmatize the words (the, a, um, etc.)
    train_df["preprocessed"] = train_df["cleaned"].apply(preprocess_review)
    test_df["preprocessed"] = test_df["cleaned"].apply(preprocess_review)

    avg_before_clean = train_df["review"].str.len().mean() # ex. "This is a bad review?!?!?!" -> 10
    avg_after_clean = train_df["cleaned"].str.len().mean()  # ex. "this is a bad review" -> lower punctuation and no special characters
    avg_before_pre = avg_after_clean  # must equal after cleaning
    avg_after_pre = train_df["preprocessed"].str.len().mean()  # ex. "bad review" -> removed stopwords and lemmatized

    # 4. Feature Extraction - Convert the text into a matrix of TF-IDF features (words to values)
    vectorizer, X_train, X_test = extract_features(
        train_df["preprocessed"].tolist(), test_df["preprocessed"].tolist()
    )

    print_data_and_features(
        stats,
        avg_before_clean,
        avg_after_clean,
        avg_before_pre,
        avg_after_pre,
        X_train.shape, 
        # Since X-train is the matrix of TF-IDF features, \
        # the shape is (number of reviews, number of unique words) - 200k * 80% of training
    )

    # 5. Models (Perceptron, Logistic Regression, Naive Bayes)
    y_train = train_df["label"]
    y_test = test_df["label"]
    for name in ["Perceptron", "Logistic Regression", "Naive Bayes"]:
        model = train_model(name, X_train, y_train) # train the model on the training data
        train_metrics = evaluate_model(model, X_train, y_train) # evaluate the model on the training data
        test_metrics = evaluate_model(model, X_test, y_test) # evaluate the model on the test data
        print_model_performance(name, train_metrics, test_metrics)

    # 6. Bigram Model & Smoothing (Add-k)
    # Use cleaned text (not preprocessed); skip empty; sample within each split
    lm_train = (
        train_df.loc[train_df["cleaned"] != "", "cleaned"]
        .sample(n=lm_train_size, replace=False, random_state=42)
        .tolist()
    )
    lm_test = (
        test_df.loc[test_df["cleaned"] != "", "cleaned"]
        .sample(n=lm_test_size, replace=False, random_state=42)
        .tolist()
    )
    bigram_model = fit_bigram(lm_train)
    eval_k1 = evaluate_bigram(bigram_model, lm_test, k=1.0)
    eval_k01 = evaluate_bigram(bigram_model, lm_test, k=0.1)
    eval_k001 = evaluate_bigram(bigram_model, lm_test, k=0.01)
    print_bigram_model_and_smoothing(bigram_model, eval_k1, eval_k01, eval_k001)

    return None


if __name__ == "__main__":
    run_experiment()
