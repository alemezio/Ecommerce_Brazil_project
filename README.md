# 📦 Predicting e-commerce delivery times. Olist dataset (Brazil)

Originated as the **final project** of the *Postgraduate Diploma in Data Science, AI and their Applications in Economics and Business*, Universidad Nacional de Córdoba ([diplocienciadedatos.com.ar](https://diplocienciadedatos.com.ar)) · 2025. This repository evolves independently from that course submission.

**Author:** Alejandro Mezio

## 🎯 The problem

[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) is the largest marketplace in Brazil. Its delivery date estimator is very imprecise: it misses by **more than 12 days** on average. A poor estimate hurts the buying experience and the sellers' reviews.

**Goal:** build a Machine Learning model that predicts the actual delivery days of an order more precisely than Olist's original estimator.

## 🏆 Headline result

**XGBoost predicts the actual delivery days with an MAE of 4.43 days in 5-fold cross-validation (4.38 on the held-out test set): a 22% error reduction over the strongest trivial baseline and 9% over linear regression.**

Test-set comparison:

| Model | MAE (days) | RMSE (days) |
|---|---|---|
| **XGBoost (selected model)** | **4.38** | **7.13** |
| Linear Regression (scientific baseline) | 4.82 | 7.16 |
| Constant prediction (train median, 9 days) | 5.60 | 8.63 |
| Olist's original estimator (business benchmark) | 12.63 | 14.53 |

An honest reading of that last row: Olist's estimator is a **promise date, not a prediction**. It overshoots actual deliveries by **+11.5 days on average**, and 93% of orders arrive on or before it, so most of its 12.6-day MAE is deliberate bias rather than inability to predict. Beating it confirms the model is useful to the business; the modelling contribution itself is measured against the constant and linear baselines above.

The most relevant features (SHAP, top 5 carrying 83% of mean |SHAP|): whether the order stays **within the same state**, the **distance in km between buyer and seller**, the **freight value**, the **chargeable weight** and the **price**.

All baseline numbers are reproducible with [`scripts/baseline_check.py`](scripts/baseline_check.py), which replicates the notebook's exact split and pipeline (its linear-regression CV matches the notebook's stored 4.824247 ± 0.034458 to six decimals).

## 🔬 Methodology

### 1. Exploratory Analysis and Cleaning ([`Part1_Olist_Exploratory_Analysis.ipynb`](Part1_Olist_Exploratory_Analysis.ipynb))

- Individual EDA of the 9 tables of the dataset (~100k orders, ~1M geolocation records): handling of duplicates, missing values and outliers.
- Join of the tables into a single order-level dataset.
- **Feature engineering:** volumetric weight, seller-customer distance (geopy), temporal variables, state-to-state routes, product categories.
- Univariate, bivariate and geographic analysis.

| Customers | Sellers | Routes |
|---|---|---|
| ![Customer distribution in Brazil](Brasil_customers.png) | ![Seller distribution in Brazil](Brasil_sellers.png) | ![Buyer-seller routes](Brasil_rutas.png) |

*The concentration of sellers in São Paulo vs. the dispersion of customers across the whole country explains why distance is the most predictive variable.*

### 2. Modelling ([`Part2_Olist_Machine_Learning.ipynb`](Part2_Olist_Machine_Learning.ipynb))

**9 regression models** were trained and compared using scikit-learn pipelines, cross-validation and hyperparameter search (Grid/Randomized Search). Validation results (5-fold CV on the training set, best configuration per model):

| Model | MAE (CV) ± fold std | RMSE (CV) |
|---|---|---|
| Constant baseline (fold-train median) | 5.635 ± 0.070 | 8.645 |
| Linear Regression (base) | 4.824 ± 0.034 | 7.148 |
| Lasso (L1) | 4.824 ± 0.035 | 7.149 |
| Ridge (L2) | 4.823 ± 0.035 | 7.149 |
| Ridge + polynomials (degree 2) | 4.781 ± 0.033 | 7.117 |
| Decision Tree | 4.756 ± 0.038 | 7.106 |
| Random Forest | 4.650 ± 0.034 | **6.985** |
| **XGBoost** | **4.428** ± n/r | 7.152 |
| SVM Regressor (LinearSVR) | 4.633 ± n/r | 7.344 |
| SGD Regressor | 4.633 ± n/r | 7.344 |

How to read the uncertainty: XGBoost's MAE lead over Random Forest (0.22 days) is roughly 6 times the fold-to-fold std (0.033-0.038 days for every model where it was persisted), so the ranking is stable. The differences within the linear family (4.824 vs 4.823) sit far inside fold noise: those are ties. Random Forest wins RMSE, but **XGBoost** was selected by MAE, the metric declared before the comparison. The model was then interpreted through feature importance and per-feature performance analysis (SHAP). *n/r: fold std not persisted by the original search run; `scripts/baseline_check.py` can recompute it.*

### Evaluation notes

- **Leakage discipline:** `review_score` exists in the dataset but is deliberately excluded from the features: reviews are written after delivery, so including it would leak the outcome into the prediction.
- *