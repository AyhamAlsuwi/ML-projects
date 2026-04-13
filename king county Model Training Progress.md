# Model Training & Iteration Log
This document details the data science workflow, feature engineering, and hyperparameter tuning process used to create the final King County Housing Price prediction model.

---

## 1. Data Cleaning & Feature Selection
Initial **Exploratory Data Analysis (EDA)** revealed a highly skewed target variable (`price`) and significant geographical clusters.

* **Dropped Features:** `id`, `grade`, and `zipcode` were removed early in the process to prevent severe overfitting.
* **Outliers:** Isolated using `IsolationForest` (contamination=0.002). Logical outliers were kept as they represented actual luxury housing market data rather than data entry errors.

---

## 2. Feature Engineering & Custom Transformers
To maximize predictive power, specific columns were processed through custom Scikit-Learn transformers:

* **DateYearExtractor:** Stripped exact timestamps and extracted just the year sold to capture macroeconomic trends.
* **SimpleRatioTransformer:** Engineered ratio features (e.g., `sqft_lot` vs `sqft_lot15`) to capture property renovations and expansions over time.
* **Geographical Clustering:** Implemented a `KMeans` clustering pipeline ($n\_clusters=37$) on `lat` and `long` coordinates to group neighborhoods.

---

## 3. Model Iterations
The preprocessed data was fed into multiple algorithms to establish baselines and optimize performance.

### Iteration 1: Baselines
* **Linear Regression:** Severely underperformed. The relationships are highly non-linear.
* **Random Forest (V1):** Established a strong baseline; outperformed Linear Regression on RMSE and MAPE.

### Iteration 2: Hyperparameter Tuning (V2)
Conducted extensive tuning using `RandomizedSearchCV` and `GridSearchCV`.
* **Tuned parameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.

### Iteration 3: Geo-Cluster Optimization (V3) — **Winner**
Refined the KMeans cluster count and ran a deeper randomized search. 
> This V3 Random Forest was selected as the final production model: `king_county_housing_model_best_RF.pkl`.

### Iteration 4 & 5: Alternative Approaches
* **Target Encoding (V4):** Attempted replacing geo-clusters with target encoding on zipcode; resulted in worse generalization.
* **XGBoost:** Despite extensive tuning of learning rates and gamma, the geo-clustered Random Forest (V3) maintained superior performance.

---

## 4. Final Production Metrics
To ensure reliability, a **95% Confidence Interval** was calculated using Scipy statistical bootstrapping on unseen test set errors.

| Metric | Value | 95% Confidence Interval |
| :--- | :--- | :--- |
| **Final Test MAPE** | 0.127% | 0.134% to 0.143% |
| **Final Test RMSE** | $148,929 | $128,460 to $199,182 |

This confirms the model generalizes well to unseen King County housing data within statistically significant bounds.
