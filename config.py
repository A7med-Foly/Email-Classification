"""
config.py — Central configuration for Email Spam Classification project.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH      = os.path.join(RAW_DIR, "spam.csv")
VECTORIZER_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
BEST_MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")
RESULTS_PATH       = os.path.join(LOGS_DIR, "model_comparison.csv")

# ── Data settings ─────────────────────────────────────────────────────────────
TARGET_COLUMN  = "label"
TEXT_COLUMN    = "text"
POSITIVE_LABEL = "spam"      # label treated as the "positive" class
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

# ── Text preprocessing ────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
USE_STEMMING       = True    # PorterStemmer
USE_LEMMATIZATION  = True    # WordNetLemmatizer

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "NaiveBayes": {
        "class": "sklearn.naive_bayes.MultinomialNB",
        "params": {"alpha": 1.0},
    },
    "LogisticRegression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {"max_iter": 1000, "random_state": RANDOM_STATE},
    },
    "SVM": {
        "class": "sklearn.svm.LinearSVC",
        "params": {"max_iter": 2000, "random_state": RANDOM_STATE},
    },
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 200, "random_state": RANDOM_STATE, "n_jobs": -1},
    },
}
