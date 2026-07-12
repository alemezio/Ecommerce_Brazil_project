"""Model pipeline: features, preprocessing, estimator and data loading.

Every definition here mirrors the notebooks (Part2 feature selection and
preprocessing; Part3 XGB_BEST). Change it here and re-run the notebooks,
or change the notebooks and update here: the smoke tests in tests/ fail
when the two drift apart.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

NUMERICAL_FEATURES = [
    "price",
    "freight_value",
    "product_chargeable_weight",
    "distance_km",
    "sales_same_state",
]
CATEGORICAL_FEATURES = ["product_sales_volume", "route"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "days_to_delivered"

RANDOM_STATE = 888

# Best hyperparameters found by the seeded randomized search in Part 2
# (random_state=888; best MAE (CV) = 4.4304).
XGB_BEST = dict(
    colsample_bytree=0.9601790561111081,
    gamma=0.7547273470940448,
    learning_rate=0.01220231163108703,
    max_depth=9,
    min_child_weight=11,
    n_estimators=1304,
    reg_alpha=0.0852809795651881,
    reg_lambda=2.106841943759883,
    subsample=0.6399331361796047,
)


def load_dataset(data_root="data/processed", name="orders_final"):
    """Load the processed order-level dataset with its saved dtypes."""
    root = Path(data_root)
    dtypes = json.loads((root / f"{name}_datatypes.txt").read_text())
    datetime_cols = [c for c, t in dtypes.items() if t == "datetime64[ns]"]
    for c in datetime_cols:
        del dtypes[c]
    return pd.read_csv(root / f"{name}.csv", dtype=dtypes,
                       parse_dates=datetime_cols)


def train_test_frames(df):
    """The notebook's exact split: 80/20, random_state=888."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]),
         NUMERICAL_FEATURES),
        ("cat", Pipeline(steps=[("onehot",
                                 OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL_FEATURES),
    ])


def make_model(**overrides):
    """The selected model: preprocessing + XGBoost with the frozen params."""
    params = {**XGB_BEST, **overrides}
    return Pipeline(steps=[
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(objective="reg:absoluteerror",
                               tree_method="hist",
                               random_state=RANDOM_STATE,
                               n_jobs=-1, **params)),
    ])
