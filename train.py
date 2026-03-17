"""
train.py
─────────────────────────────────────────────────────────────────────────────
End-to-end training pipeline for Email Spam Classification.

Usage
-----
    python train.py

Steps
-----
  1. Load & clean spam.csv
  2. Encode labels (spam=1, ham=0)
  3. Text preprocessing (lower → tokenize → stopwords → stem → lemmatize)
  4. TF-IDF vectorization (fit on train only)
  5. Train all models: Naive Bayes, Logistic Regression, SVM, Random Forest
  6. Evaluate & print comparison table
  7. Save best model → models/best_model.pkl
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BEST_MODEL_PATH, LABEL_ENCODER_PATH, LOGS_DIR, MODEL_CONFIGS,
    MODELS_DIR, RANDOM_STATE, RAW_DATA_PATH, RESULTS_PATH,
    TARGET_COLUMN, TEXT_COLUMN, TEST_SIZE, TFIDF_MAX_FEATURES,
    USE_LEMMATIZATION, USE_STEMMING, VECTORIZER_PATH,
)
from src.data_loader import clean_data, encode_labels, load_data, split_data
from src.logger import setup_logger
from src.model_evaluation import (
    evaluate_all_models, print_results, save_results, select_best_model,
)
from src.model_training import save_model, train_all_models
from src.text_preprocessing import (
    build_vectorizer, download_nltk_resources,
    fit_transform_tfidf, preprocess_series, save_vectorizer,
)


def main() -> None:
    setup_logger(log_dir=LOGS_DIR)
    import logging
    log = logging.getLogger(__name__)
    log.info("═" * 60)
    log.info("  EMAIL SPAM CLASSIFICATION — TRAINING PIPELINE")
    log.info("═" * 60)

    # 1 ─ NLTK resources
    log.info("Step 1/6 — Downloading NLTK resources …")
    download_nltk_resources()

    # 2 ─ Load & clean
    log.info("Step 2/6 — Loading & cleaning data …")
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df, label_encoder = encode_labels(df, TARGET_COLUMN, LABEL_ENCODER_PATH)

    # 3 ─ Split (on raw text, before vectorisation)
    log.info("Step 3/6 — Splitting data …")
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df, TEXT_COLUMN, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )

    # 4 ─ Text preprocessing
    log.info("Step 4/6 — Preprocessing text …")
    X_train_clean = preprocess_series(X_train_raw, USE_STEMMING, USE_LEMMATIZATION)
    X_test_clean  = preprocess_series(X_test_raw,  USE_STEMMING, USE_LEMMATIZATION)

    # 5 ─ TF-IDF
    log.info("Step 5/6 — Vectorizing with TF-IDF (max_features=%d) …", TFIDF_MAX_FEATURES)
    vectorizer = build_vectorizer(TFIDF_MAX_FEATURES)
    X_train, X_test = fit_transform_tfidf(vectorizer, X_train_clean, X_test_clean)
    save_vectorizer(vectorizer, VECTORIZER_PATH)

    # 6 ─ Train
    log.info("Step 6/6 — Training %d models …", len(MODEL_CONFIGS))
    trained = train_all_models(MODEL_CONFIGS, X_train, y_train, MODELS_DIR)

    # Evaluate
    results_df = evaluate_all_models(trained, X_test, y_test)
    print_results(results_df)
    save_results(results_df, RESULTS_PATH)

    # Save best
    best_name, best_model = select_best_model(trained, X_test, y_test)
    save_model(best_model, BEST_MODEL_PATH)
    log.info("✓ Best model '%s' saved to '%s'.", best_name, BEST_MODEL_PATH)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
