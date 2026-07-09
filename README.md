# 📦 Predicción de días de entrega en e-commerce. Dataset Olist (Brasil)

**Proyecto final** de la *Diplomatura en Ciencia de Datos, IA y sus Aplicaciones en Economía y Negocios*, Universidad Nacional de Córdoba ([diplocienciadedatos.com.ar](https://diplocienciadedatos.com.ar)) · Grupo 8 · 2025

## 🎯 El problema

[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) es el marketplace más grande de Brasil. Su estimador de fecha de entrega es muy impreciso: se equivoca en promedio **más de 12 días**. Una estimación pobre afecta la experiencia de compra y las reviews de los vendedores.

**Objetivo:** construir un modelo de Machine Learning que prediga los días reales de entrega de una orden con mayor precisión que el estimador original de Olist.

## 🏆 Resultado principal

El modelo seleccionado (**XGBoost**) reduce el error promedio de estimación de **12.6 a 4.4 días** sobre el conjunto de test:

| Modelo | MAE (días) | RMSE (días) |
|---|---|---|
| **XGBoost (nuestro modelo)** | **4.38** | **7.13** |
| Regresión Lineal (baseline) | 4.82 | 7.16 |
| Estimador original de Olist | 12.63 | 14.53 |

Las variables más relevantes para la predicción resultaron ser la **distancia en km entre comprador y vendedor**, si la orden es **dentro del mismo estado**, el **valor del flete** y la **ruta comprador-vendedor**.

## 🔬 Metodología

### 1. Análisis Exploratorio y Limpieza ([`Parte1_Olist_Analisis_Exploratorio.ipynb`](Parte1_Olist_Analisis_Exploratorio.ipynb))

- EDA individual de las 9 tablas del dataset (~100k órdenes, ~1M registros de geolocalización): tratamiento de duplicados, nulos y outliers.
- Unión de las tablas en un dataset único a nivel orden.
- **Feature engineering:** peso volumétrico, distancia seller-customer (geopy), variables temporales, rutas entre estados, categorías de producto.
- Análisis univariado, bivariado y geográfico.

| Clientes | Vendedores | Rutas |
|---|---|---|
| ![Distribución de clientes en Brasil](Brasil_customers.png) | ![Distribución de vendedores en Brasil](Brasil_sellers.png) | ![Rutas comprador-vendedor](Brasil_rutas.png) |

*La concentración de vendedores en San Pablo vs. la dispersión de clientes por todo el país explica por qué la distancia es la variable más predictiva.*

### 2. Modelado ([`Parte2_Olist_Machine_Learning.ipynb`](Parte2_Olist_Machine_Learning.ipynb))

Se entrenaron y compararon **9 modelos de regresión** con pipelines de scikit-learn, validación cruzada y búsqueda de hiperparámetros (Grid/Randomized Search). Resultados en validación:

| Modelo | MAE (CV) | RMSE (CV) |
|---|---|---|
| Regresión Lineal (base) | 4.824 | 7.148 |
| Lasso (L1) | 4.824 | 7.149 |
| Ridge (L2) | 4.823 | 7.149 |
| Ridge + polinómicas (grado 2) | 4.781 | 7.117 |
| Decision Tree | 4.756 | 7.106 |
| Random Forest | 4.650 | **6.985** |
| **XGBoost** | **4.428** | 7.152 |
| SVM Regressor (LinearSVR) | 4.633 | 7.344 |
| SGD Regressor | 4.633 | 7.344 |

Se seleccionó **XGBoost** por su menor MAE, y se interpretó el modelo con importancia de variables y análisis de performance por feature (SHAP).

## 📁 Estructura del repositorio

| Carpeta/Archivo | Descripción |
|---|---|
| `Parte1_Olist_Analisis_Exploratorio.ipynb` | EDA, limpieza, unión de tablas y feature engineering. |
| `Parte2_Olist_Machine_Learning.ipynb` | Pipelines, entrenamiento, comparación y selección de modelos. |
| `bbdd_limpia/` | Esquema de tipos del dataset final. El CSV (`dataset_final_agrupado.csv`) no se versiona: lo regenera `Parte1` a partir de los datos crudos. |
| `Brasil_*.png` | Mapas de clientes, vendedores y rutas. |
| `requirements.txt` | Dependencias (Python 3.11). |

## ⚙️ Instalación y reproducción

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/alemezio/diplodatos_proyecto_final.git
   cd diplodatos_proyecto_final
   ```

2. Instalar dependencias (Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar los notebooks en orden:

   - `Parte1_Olist_Analisis_Exploratorio.ipynb`: Descarga los datos crudos vía `kagglehub` y genera el dataset limpio.
   - `Parte2_Olist_Machine_Learning.ipynb`: Entrena y evalúa los modelos a partir de `bbdd_limpia/dataset_final_agrupado.csv`.

## 🔮 Trabajo futuro

- **Incorporar reviews:** analizar la relación entre calificaciones y tiempos de entrega para construir un "score" de vendedor.
- **Acotar el rango temporal:** entrenar solo con datos de 2018, donde el volumen de ventas era mayor y más estable.

## 📄 Datos y licencias

Los datos provienen del [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), publicado por Olist bajo licencia **CC BY-NC-SA 4.0**: ~100k órdenes reales (2016-2018), anonimizadas. Los datasets derivados conservan esa licencia. El código de este repositorio se distribuye bajo licencia [MIT](LICENSE).

Este proyecto naci