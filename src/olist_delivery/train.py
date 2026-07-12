"""Train the delivery-time model and persist it with its metrics.

Reproduces the notebook protocol exactly: 80/20 split (random_state=888),
fit the frozen XGBoost configuration on the training part, evaluate once
on the held-out test part. Expected test metrics (up to small XGBoost
threading jitter, ~0.01 days): MAE 4.38, RMSE 7.14.

Usage:
    python -m src.olist_delivery.train [--data-root data/processed]
        [--output models/delivery_model.joblib] [--fast]

--fast trains a 20-tree model for plumbing checks only (used by the smoke
tests); its metrics are not meaningful.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .pipeline import load_dataset, make_model, train_test_frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="models/delivery_model.joblib")
    parser.add_argument("--metrics-out",
                        default="models/delivery_model_metrics.json")
    parser.add_argument("--fast", action="store_true",
                        help="20-tree model for plumbing checks only")
    args = parser.parse_args(argv)

    df = load_dataset(args.data_root)
    X_train, X_test, y_train, y_test = train_test_frames(df)
    print(f"train: {len(X_train)} orders | test: {len(X_test)} orders")

    model = make_model(n_estimators=20) if args.fast else make_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred = model.predict(X_test)
    metrics = {
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(root_mean_squared_error(y_test, pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fast_mode": args.fast,
        "fit_seconds": round(elapsed, 1),
    }
    print(f"test MAE: {metrics['test_mae']:.4f} | "
          f"test RMSE: {metrics['test_rmse']:.4f} | "
          f"fit: {elapsed:.0f}s{' (fast mode)' if args.fast else ''}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print(f"model saved to {out} | metrics saved to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
