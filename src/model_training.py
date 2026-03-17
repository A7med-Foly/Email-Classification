"""
src/model_training.py
Instantiate, train, save and load classifiers.
"""

import importlib
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)


def instantiate_model(class_path: str, params: dict):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**params)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    logger.info("Trained %s.", type(model).__name__)
    return model


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to '%s'.", path)


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from '%s'.", path)
    return model


def train_all_models(
    model_configs: dict,
    X_train,
    y_train,
    models_dir: str,
) -> dict:
    trained = {}
    for name, cfg in model_configs.items():
        logger.info("── Training %s ──", name)
        model = instantiate_model(cfg["class"], cfg.get("params", {}))
        model = train_model(model, X_train, y_train)
        save_model(model, os.path.join(models_dir, f"{name}.pkl"))
        trained[name] = model
    logger.info("All %d models trained.", len(trained))
    return trained
