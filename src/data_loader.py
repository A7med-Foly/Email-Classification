"""
src/data_loader.py
Load and clean the raw spam.csv dataset.
"""

import logging
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download spam.csv from:\n"
            "  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n"
            "and place it in data/raw/spam.csv"
        )
    df = pd.read_csv(path, encoding="latin1")
    logger.info("Loaded %d rows from '%s'.", len(df), path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed filler columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info("Dropped columns: %s", unnamed)

    # Rename v1/v2 if present
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
        logger.info("Renamed v1→label, v2→text.")

    df = df.dropna(subset=["label", "text"]).drop_duplicates().reset_index(drop=True)
    logger.info("Clean shape: %s | spam=%d | ham=%d",
                df.shape,
                (df["label"] == "spam").sum(),
                (df["label"] == "ham").sum())
    return df


def encode_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    encoder_save_path: str | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode spam/ham → 1/0. Returns updated df and fitted encoder."""
    le = LabelEncoder()
    df = df.copy()
    df[label_col] = le.fit_transform(df[label_col])
    logger.info("Label encoding: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    if encoder_save_path:
        os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
        with open(encoder_save_path, "wb") as f:
            pickle.dump(le, f)
        logger.info("Label encoder saved to '%s'.", encoder_save_path)

    return df, le


def load_label_encoder(path: str) -> LabelEncoder:
    with open(path, "rb") as f:
        le = pickle.load(f)
    return le


def split_data(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
