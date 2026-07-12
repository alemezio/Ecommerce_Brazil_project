"""Smoke tests: the src/ pipeline must not drift from the notebooks.

Fast tests run on the committed 2000-row sample (data/sample/) and need no
full dataset. Tests marked as full-data are skipped unless
data/processed/orders_final.csv exists (regenerate it with Part1), and
checkpoint tests are skipped unless the trained model artifacts exist
(train with `python -m src.olist_delivery.train`).

Checkpoints asserted against the notebooks:
- split sizes 72109 / 18028 and the first test indices (random_state=888)
- trained-model test MAE 4.3822 (Part2's stored output; tolerance covers
  XGBoost hist threading jitter)
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.olist_delivery.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    load_dataset,
    make_model,
    train_test_frames,
)

ROOT = Path(__file__).resolve().parent.parent
FULL_DATA = ROOT / "data/processed/orders_final.csv"
MODEL = ROOT / "models/delivery_model.joblib"
METRICS = ROOT / "models/delivery_model_metrics.json"

# Notebook checkpoints (Part2, random_state=888)
EXPECTED_SPLIT = (72109, 18028)
EXPECTED_FIRST_TEST_INDICES = [4315, 3014, 35953, 15236, 65412, 78219,
                               13313, 15520]
EXPECTED_TEST_MAE = 4.3822
MAE_TOLERANCE = 0.05  # generous cover for hist threading jitter


@pytest.fixture(scope="module")
def sample():
    return load_dataset(ROOT / "data/sample", "orders_sample")


def test_sample_has_the_model_contract(sample):
    assert len(sample) == 2000
    for col in FEATURES + [TARGET]:
        assert col in sample.columns, f"missing column {col}"
    assert set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) == set(FEATURES)


def test_fast_train_predict_roundtrip(sample):
    model = make_model(n_estimators=5)
    X, y = sample[FEATURES], sample[TARGET]
    model.fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert (pred > 0).all() and (pred < 200).all()


def test_unseen_categories_warn_not_fail(sample, capsys):
    from src.olist_delivery.predict import warn_unseen_categories
    model = make_model(n_estimators=5)
    model.fit(sample[FEATURES], sample[TARGET])
    X = sample[FEATURES].head(5).copy()
    X.loc[X.index[0], "route"] = "ZZ-ZZ"
    warn_unseen_categories(model, X)
    assert "ZZ-ZZ" in capsys.readouterr().err
    pred = model.predict(X)  # must not raise
    assert len(pred) == 5


@pytest.mark.skipif(not FULL_DATA.exists(),
                    reason="full dataset not present (run Part1)")
def test_split_reproduces_the_notebook():
    df = load_dataset(ROOT / "data/processed")
    X_train, X_test, y_train, y_test = train_test_frames(df)
    assert (len(X_train), len(X_test)) == EXPECTED_SPLIT
    assert list(X_test.index[:8]) == EXPECTED_FIRST_TEST_INDICES


@pytest.mark.skipif(not (MODEL.exists() and METRICS.exists()),
                    reason="trained model not present (run train CLI)")
def test_trained_model_matches_notebook_checkpoint():
    metrics = json.loads(METRICS.read_text())
    assert not metrics["fast_mode"], "committed metrics come from --fast"
    assert abs(metrics["test_mae"] - EXPECTED_TEST_MAE) < MAE_TOLERANCE, (
        f"test MAE {metrics['test_mae']:.4f} drifted from the notebook "
        f"checkpoint {EXPECTED_TEST_MAE}")
    assert (metrics["n_train"], metrics["n_test"]) == EXPECTED_SPLIT


@pytest.mark.skipif(not MODEL.exists(),
                    reason="trained model not present (run train CLI)")
def test_saved_model_predicts_on_sample(sample):
    """Sanity of the persisted artifact in THIS environment.

    A failure here with wildly wrong predictions (e.g. negative days)
    usually means the model was pickled under a different XGBoost version
    than the one installed: the artifact loads but predicts garbage.
    Re-train in the current environment or align the xgboost version.
    """
    import joblib
    model = joblib.load(MODEL)
    pred = model.predict(sample[FEATURES].head(50))
    assert len(pred) == 50
    assert (pred > 0).all() and (pred < 200).all(), (
        f"implausible predictions (min {pred.min():.2f}); likely an "
        f"xgboost version mismatch with the saved artifact")
