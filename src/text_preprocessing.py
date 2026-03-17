"""
src/text_preprocessing.py
All NLP preprocessing steps: clean → tokenize → stopwords → stem → lemmatize → TF-IDF.
"""

import logging
import os
import pickle
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── NLTK resource download (silent) ───────────────────────────────────────────

def download_nltk_resources() -> None:
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}" if "punkt" in r else
                           f"corpora/{r}")
        except LookupError:
            nltk.download(r, quiet=True)
    logger.info("NLTK resources ready.")


# ── Individual steps ───────────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def remove_special_chars_and_numbers(tokens: list[str]) -> list[str]:
    return [
        t for t in tokens
        if t not in string.punctuation and not t.isdigit()
    ]


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = set(stopwords.words("english"))
    return [t for t in tokens if t not in sw]


def stem(tokens: list[str]) -> list[str]:
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]


def lemmatize(tokens: list[str]) -> list[str]:
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]


# ── Full pipeline on a single text string ─────────────────────────────────────

def preprocess_text(
    text: str,
    use_stemming: bool = True,
    use_lemmatization: bool = True,
) -> str:
    """
    Apply the full NLP pipeline to one text string.
    Returns a clean space-joined token string ready for TF-IDF.
    """
    text   = to_lowercase(text)
    tokens = tokenize(text)
    tokens = remove_special_chars_and_numbers(tokens)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem(tokens)
    if use_lemmatization:
        tokens = lemmatize(tokens)
    return " ".join(tokens)


# ── Apply to a full DataFrame ─────────────────────────────────────────────────

def preprocess_series(
    series: pd.Series,
    use_stemming: bool = True,
    use_lemmatization: bool = True,
) -> pd.Series:
    logger.info("Preprocessing %d texts …", len(series))
    processed = series.apply(
        lambda t: preprocess_text(t, use_stemming, use_lemmatization)
    )
    logger.info("Text preprocessing complete.")
    return processed


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features)


def fit_transform_tfidf(
    vectorizer: TfidfVectorizer,
    train_texts: pd.Series,
    test_texts: pd.Series,
):
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    logger.info(
        "TF-IDF shapes — train: %s | test: %s",
        X_train.shape, X_test.shape,
    )
    return X_train, X_test


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Vectorizer saved to '%s'.", path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, "rb") as f:
        v = pickle.load(f)
    logger.info("Vectorizer loaded from '%s'.", path)
    return v
