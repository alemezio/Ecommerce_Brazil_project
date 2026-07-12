"""Predict delivery times for new orders from the command line.

Reads a CSV containing the seven feature columns, writes the same rows
plus a `predicted_delivery_days` column.

Usage:
    python -m src.olist_delivery.predict orders.csv
        [--model models/delivery_model.joblib] [--output predictions.csv]

Categorical values never seen in training (e.g. a new route label) are
handled by the encoder as all-zeros, which silently falls back to the
model's baseline behavior; a warning listing them is printed so the
fallback is never silent.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from .pipeline import CATEGORICAL_FEATURES, FEATURES


def warn_unseen_categories(model, X):
    """Print a warning for category values the training data never saw."""
    encoder = (model.named_steps["preprocessor"]
               .named_transformers_["cat"].named_steps["onehot"])
    for col, known in zip(CATEGORICAL_FEATURES, encoder.categories_):
        unseen = set(X[col].dropna().unique()) - set(known)
        if unseen:
            print(f"WARNING: column '{col}' contains values unseen in "
                  f"training: {sorted(unseen)}. They are encoded as "
                  f"all-zeros (baseline behavior).", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with the feature columns")
    parser.add_argument("--model", default="models/delivery_model.joblib")
    parser.add_argument("--output", default=None,
                        help="output CSV (default: <input>_predictions.csv)")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: input is missing required columns: {missing}",
              file=sys.stderr)
        return 1

    model = joblib.load(args.model)
    X = df[FEATURES]
    warn_unseen_categories(model, X)

    out_df = df.copy()
    out_df["predicted_delivery_days"] = model.predict(X)

    out_path = Path(args.output) if args.output else (
        Path(args.input).with_name(Path(args.input).stem
                                   + "_predictions.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"{len(out_df)} predictions written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
