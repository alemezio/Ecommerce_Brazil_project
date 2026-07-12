"""Olist delivery-time prediction: pipeline, training and inference.

This package is the single source of truth for the model pipeline outside
the notebooks. It mirrors Part2/Part3 exactly: same features, same
preprocessing, same split, same hyperparameters. The smoke tests assert
that this equivalence holds.
"""

from .pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    XGB_BEST,
    load_dataset,
    make_model,
    make_preprocessor,
    train_test_frames,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "TARGET",
    "XGB_BEST",
    "load_dataset",
    "make_model",
    "make_preprocessor",
    "train_test_frames",
]
