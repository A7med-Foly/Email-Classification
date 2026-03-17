"""
src/predict.py
Inference: raw email text → spam / ham prediction.
"""

import logging

import numpy as np

from src.text_preprocessing import load_vectorizer, preprocess_text
from src.model_training import load_model
from src.data_loader import load_label_encoder

logger = logging.getLogger(__name__)


def load_artifacts(vectorizer_path: str, model_path: str, encoder_path: str):
    vectorizer = load_vectorizer(vectorizer_path)
    model      = load_model(model_path)
    encoder    = load_label_encoder(encoder_path)
    return vectorizer, model, encoder


def predict(
    texts: list[str] | str,
    vectorizer,
    model,
    encoder,
    use_stemming: bool = True,
    use_lemmatization: bool = True,
) -> list[dict]:
    """
    Predict spam/ham for one or more raw email texts.
    Returns a list of dicts: {"text", "label", "confidence"}.
    """
    if isinstance(texts, str):
        texts = [texts]

    processed = [preprocess_text(t, use_stemming, use_lemmatization) for t in texts]
    X         = vectorizer.transform(processed)
    preds     = model.predict(X)
    labels    = encoder.inverse_transform(preds)

    # Confidence: use predict_proba if available, else decision_function
    confidences = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        confidences = probs.max(axis=1).tolist()
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X)
        if df_vals.ndim == 1:
            import scipy.special
            confidences = scipy.special.expit(df_vals).tolist()
        else:
            confidences = df_vals.max(axis=1).tolist()
    else:
        confidences = [None] * len(texts)

    return [
        {"text": t, "label": lbl, "confidence": conf}
        for t, lbl, conf in zip(texts, labels, confidences)
    ]


def predict_from_paths(
    texts: list[str] | str,
    vectorizer_path: str,
    model_path: str,
    encoder_path: str,
) -> list[dict]:
    vectorizer, model, encoder = load_artifacts(vectorizer_path, model_path, encoder_path)
    return predict(texts, vectorizer, model, encoder)
