# 📦 Predicting e-commerce delivery times. Olist dataset (Brazil)

Originated as the **final project** of the *Postgraduate Diploma in Data Science, AI and their Applications in Economics and Business*, Universidad Nacional de Córdoba ([diplocienciadedatos.com.ar](https://diplocienciadedatos.com.ar)) · 2025. This repository evolves independently from that course submission.

**Author:** Alejandro Mezio

## 🎯 The problem

[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) is the largest marketplace in Brazil. Its delivery date estimator is very imprecise: it misses by **more than 12 days** on average. A poor estimate hurts the buying experience and the sellers' reviews.

**Goal:** build a Machine Learning model that predicts the actual delivery days of an order more precisely than Olist's original estimator.

## 🏆 Headline result

The selected model (**XGBoost**) reduces the average estimation error from **12.6 to 4.4 days** on the test set:

| Model | MAE (days) | RMSE (days) |
|---|---|---|
| **XGBoost (our model)** | **4.38** | **7.13** |
| Linear Regression (baseline) | 4.82 | 7.16 |
| Olist's original estimator | 12.63 | 14.53 |

The most relevant features for the prediction turned out to be the **distance in km between buyer and seller**, whether the order stays **within the same state**, the **freight value** and the **buyer-seller route**.

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

**9 regression models** were trained and compared using scikit-learn pipelines, cross-validation and hyperparameter search (Grid/Randomized Search). Validation results:

| Model | MAE (CV) | RMSE (CV) |
|---|---|---|
| Linear Regression (base) | 4.824 | 7.148 |
| Lasso (L1) | 4.824 | 7.149 |
| Ridge (L2) | 4.823 | 7.149 |
| Ridge + polynomials (degree 2) | 4.781 | 7.117 |
| Decision Tree | 4.756 | 7.106 |
| Random Forest | 4.650 | **6.985** |
| **XGBoost** | **4.428** | 7.152 |
| SVM Regressor (LinearSVR) | 4.633 | 7.344 |
| SGD Regressor | 4.633 | 7.344 |

**XGBoost** was selected for its lowest MAE, and the model was interpreted through feature importance and per-feature performance analysis (SHAP).

## 📁 Repository structure

| Folder/File | Description |
|---|---|
| `Part1_Olist_Exploratory_Analysis.ipynb` | EDA, cleaning, table joins and feature engineering. |
| `Part2_Olist_Machine_Learning.ipynb` | Pipelines, training, model comparison and selection. |
| `bbdd_limpia/` | Datatype schema of the final dataset. The CSV (`dataset_final_agrupado.csv`) is not versioned: `Part1` regenerates it from the raw data. |
| `Brasil_*.png` | Maps of customers, sellers and routes. |
| `requirements.txt` | Dependencies (Python 3.11). |

## ⚙️ Installation and reproduction

1. Clone the repository:

   ```bash
   git clone https://github.com/alemezio/Olist_Ecommerce_Brazil_project.git
   cd Olist_Ecommerce_Brazil_project
   ```

2. Install dependencies (Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in order:

   - `Part1_Olist_Exploratory_Analysis.ipynb`: Downloads the raw data via `kagglehub` and generates the clean dataset.
   - `Part2_Olist_Machine_Learning.ipynb`: Trains and evaluates the models from `bbdd_limpia/dataset_final_agrupado.csv`.

## 🔮 Future work

- **Incorporate reviews:** analyze the relation between ratings and delivery times to build a seller "score".
- **Narrow the temporal range:** train only with 2018 data, when the sales volume was higher and more stable.

## 📄 Data and licenses

The data comes from the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), published by Olist under the **CC BY-NC-SA 4.0** license: ~100k real, anonymized orders (2016-2018). Derived datasets keep that license. The code in this repository is distributed under the [MIT](LICENSE) license.

## 📬 Contact

Questions or suggestions:

- 💼 [LinkedIn](https://www.linkedin.com/in/alejandro-mezio/)
- 📧 [alejand