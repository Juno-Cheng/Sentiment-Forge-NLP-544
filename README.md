# Sentiment-Forge-NLP-544

> End-to-end sentiment analysis pipeline for Amazon Office Products reviews. TF-IDF features, three classifiers, and a smoothed bigram language model. Built for CSCI 544.

---

## Overview

**Sentiment Forge** turns raw Amazon product reviews into sentiment predictions and a smoothed bigram language model. The pipeline covers the full NLP workflow:

- Text cleaning and preprocessing (HTML, URLs, contractions, stopwords, lemmatization)
- TF-IDF feature extraction
- Training and evaluating three classifiers
- Building a bigram language model with add-k smoothing and corpus-level perplexity

Built as **HW1 for CSCI 544 (Fall 2026)**.

---

## Pipeline

| Stage | What Happens |
|---|---|
| **1. Data Prep** | Load TSV, clean ratings, map 4–5 → positive and 1–2 → negative, drop 3s, sample 100k per class, 80/20 stratified split |
| **2. Cleaning** | Unescape HTML, lowercase, strip tags/URLs, fix curly apostrophes, expand contractions, keep only letters and whitespace |
| **3. Preprocessing** | Remove NLTK stopwords (keep `no`, `nor`, `not`), WordNet lemmatize, rejoin |
| **4. Features** | Default `TfidfVectorizer`, fit on train only, transform both splits |
| **5–7. Models** | Perceptron, Logistic Regression, Multinomial Naive Bayes |
| **8. Language Model** | Bigram counts, add-k smoothing, corpus-level perplexity for k = 1.0, 0.1, 0.01 |

---

## Models

| Model | Configuration |
|---|---|
| **Perceptron** | `random_state=42, max_iter=1000, tol=1e-3` |
| **Logistic Regression** | `random_state=42, solver="lbfgs", max_iter=1000, tol=1e-4` |
| **Naive Bayes** | `MultinomialNB(alpha=1.0)` |

Each model is evaluated with **accuracy, precision, recall, and F1** on both training and test sets (binary, `pos_label=1`, `zero_division=0`).

---

## Requirements

- **Python 3.12**
- See `requirements.txt` for pinned versions

Key dependencies:

```
numpy==2.3.5
scipy==1.18.1
pandas==2.2.3
scikit-learn==1.9.1
nltk==3.10.3
contractions==0.1.73
```

Install NLTK resources once locally:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

> The grading environment provides these resources. The script must **not** download data or install packages at runtime.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/Sentiment-Forge-NLP-544.git
cd Sentiment-Forge-NLP-544

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK resources (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Place data.tsv in the project root
#    (Amazon US Office Products reviews, extracted from the archive link in the handout)
```

---

## Usage

```bash
python sentiment.py
```

This runs the full pipeline and prints **exactly 37 lines** to stdout:

1. Data and feature statistics (8 lines)
2. Classifier metrics for each model (24 lines)
3. Bigram model and smoothing comparison (5 lines)

### Programmatic Use

Each function can be called independently:

```python
from sentiment import prepare_data, clean_review, preprocess_review

train_df, test_df, stats = prepare_data("data.tsv", n_per_class=100)
cleaned = clean_review("These pens aren't reliable.")
preprocessed = preprocess_review(cleaned)
```

> Importing the module produces **no output** and does **no training**.

---

## Project Structure

```
Sentiment-Forge-NLP-544/
├── sentiment.py          # Main pipeline (submitted file)
├── requirements.txt      # Pinned dependencies
├── data.tsv              # Amazon Office Products reviews (not tracked)
└── README.md
```

---

## Function Reference

| Function | Purpose |
|---|---|
| `prepare_data(path, n_per_class=100000)` | Load, clean, sample, split → `(train_df, test_df, stats)` |
| `clean_review(text)` | HTML/URL/contraction cleanup → cleaned string |
| `preprocess_review(text)` | Stopword removal + lemmatization |
| `extract_features(train_texts, test_texts)` | Fit TF-IDF → `(vectorizer, X_train, X_test)` |
| `train_model(name, X_train, y_train)` | Fit one of the three classifiers |
| `evaluate_model(model, X, y)` | Return accuracy, precision, recall, F1 |
| `fit_bigram(train_texts, min_count=3)` | Build vocab, bigram counts, context totals |
| `evaluate_bigram(model, test_texts, k=1.0)` | Unseen count, token count, add-k perplexity |
| `run_experiment(...)` | Full pipeline, prints 37 lines |

---

## Notes

- Sparse matrices are **never densified**.
- The vectorizer is fit **only** on training data — no leakage.
- Bigram vocabulary is fit **only** on the training subset.
- Row order and indices are preserved throughout.
- Returned values are unrounded; only printed outputs are formatted to 4 decimal places.

---

## License

Coursework for **CSCI 544 — Fall 2026**. Not intended for redistribution.

---

## Topics

`nlp` · `sentiment-analysis` · `tfidf` · `bigram` · `naive-bayes` · `logistic-regression` · `perceptron` · `scikit-learn` · `nltk` · `csci-544`
