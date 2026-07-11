"""Baseline and uncertainty checks backing the README results section.

Reproduces the exact protocol of Part2_Olist_Machine_Learning.ipynb
(feature selection, preprocessing, train/test split with random_state=888,
5-fold CV) and computes:

1. Validation anchor: linear-regression CV MAE/RMSE, which must match the
   values stored in the notebook (-4.824247 +/- 0.034458) to prove the
   split and pipeline are reproduced exactly.
2. Constant-median baseline: 5-fold CV on train (median of each fold's
   training part) and test evaluation (median of the full train set).
3. Olist's original estimator (days_estimated): MAE/RMSE on the same CV
   folds and on the test set, plus its bias (mean of estimated - actual).
4. XGBoost with the best hyperparameters found by the notebook's
   RandomizedSearchCV: 5-fold CV (for the fold std the notebook did not
   print) and test metrics.

Usage: python scripts/baseline_check.py [--skip-xgb]
Writes results to scripts/baseline_check_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

NUMERICAL = ["price", "freight_value", "product_chosen_weight", "distance_km",
             "sales_same_state"]
CATEGORICAL = ["product_venta", "rutas"]
TARGETS = ["days_to_delivered", "days_estimated"]

# Best hyperparameters printed by the notebook's RandomizedSearchCV (cell
# "Best ..." under section 6. XGBoost; search re-run 2026-07-10,
# best MAE (CV) = 4.4317).
XGB_BEST = dict(
    colsample_bytree=0.9658043617192876,
    gamma=0.6323703148318477,
    learning_rate=0.039226743328387936,
    max_depth=7,
    min_child_weight=11,
    n_estimators=1012,
    reg_alpha=1.7445653986952265,
    reg_lambda=2.3230080444539265,
    subsample=0.977362924345745,
)


def load_data():
    dtypes = json.loads(
        (ROOT / "bbdd_limpia/dataset_final_agrupado_datatypes.txt").read_text())
    datetime_cols = [c for c, t in dtypes.items() if t == "datetime64[ns]"]
    for c in datetime_cols:
        del dtypes[c]
    df = pd.read_csv(ROOT / "bbdd_limpia/dataset_final_agrupado.csv",
                     dtype=dtypes, parse_dates=datetime_cols)
    X = df[NUMERICAL + CATEGORICAL]
    y = df[TARGETS]
    return train_test_split(X, y, test_size=0.2, random_state=888)


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERICAL),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL),
    ])


def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred))}


def fold_stats(maes, rmses):
    return {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "mae_folds": [float(v) for v in maes]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    yt = y_train["days_to_delivered"].to_numpy()
    ye = y_train["days_estimated"].to_numpy()
    results = {"n_train": len(X_train), "n_test": len(X_test)}

    # 1. Validation anchor: linear regression CV, must match the notebook.
    linreg = Pipeline([("preprocessor", preprocessor()),
                       ("linreg", LinearRegression())])
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error"}
    cv = cross_validate(linreg, X_train, yt, cv=5, scoring=scoring)
    results["linreg_cv"] = fold_stats(-cv["test_mae"], -cv["test_rmse"])

    # 2 & 3. Median baseline and Olist estimator on the same folds.
    med_m, med_r, est_m, est_r = [], [], [], []
    for tr, va in KFold(n_splits=5).split(X_train):
        pred = np.full(len(va), np.median(yt[tr]))
        med_m.append(mean_absolute_error(yt[va], pred))
        med_r.append(root_mean_squared_error(yt[va], pred))
        est_m.append(mean_absolute_error(yt[va], ye[va]))
        est_r.append(root_mean_squared_error(yt[va], ye[va]))
    results["median_cv"] = fold_stats(med_m, med_r)
    results["olist_estimator_cv"] = fold_stats(est_m, est_r)

    # Test-set versions.
    yt_test = y_test["days_to_delivered"].to_numpy()
    ye_test = y_test["days_estimated"].to_numpy()
    results["median_test"] = metrics(yt_test,
                                     np.full(len(yt_test), np.median(yt)))
    results["olist_estimator_test"] = metrics(yt_test, ye_test)
    results["olist_bias_mean_days_full"] = float(
        (np.concatenate([ye, ye_test]) - np.concatenate([yt, yt_test])).mean())
    results["target_mean_full"] = float(np.concatenate([yt, yt_test]).mean())

    # 4. XGBoost best config: CV fold std + test metrics.
    if not args.skip_xgb:
        from xgboost import XGBRegressor
        xgb = Pipeline([
            ("preprocessor", preprocessor()),
            ("xgb", XGBRegressor(objective="reg:absoluteerror",
                                 tree_method="hist", random_state=888,
                                 n_jobs=-1, **XGB_BEST)),
        ])
        cvx = cross_validate(xgb, X_train, yt, cv=5, scoring=scoring)
        results["xgb_cv"] = fold_stats(-cvx["test_mae"], -cvx["test_rmse"])
        xgb.fit(X_train, yt)
        results["xgb_test"] = metrics(yt_test, xgb.predict(X_test))

    out = ROOT / "scripts/baseline_check_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
