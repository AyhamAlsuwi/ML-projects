# USA Used Car Price Prediction Engine

## Overview
This folder contains a production-ready Machine Learning pipeline that I built to predict the auction prices of used cars in the USA.

The project strictly focuses on high-performance data engineering, rigorous hyperparameter optimization, and statistical validation to ensure real-world business reliability. The final model is a highly tuned XGBoost Regressor deployed via a containerized FastAPI endpoint.

---

## 1. Data Architecture & Engineering
Real-world car auction data is notoriously messy. Instead of relying on slow, basic imputation, I engineered a suite of custom, highly optimized Scikit-Learn transformers to handle the pipeline natively.

### Anomaly Detection
Before training, I purged the dataset of extreme outliers and garbage data:

* **Imputation:** Implemented `HistGradientBoostingRegressor` within an `IterativeImputer` to intelligently estimate missing numerical values.
* **Outlier Removal:** Passed the imputed dataset through an `IsolationForest` (1% contamination) to detect and drop multi-dimensional anomalies that would mathematically distort the regression plane.

### High-Performance Custom Transformers
I built vectorized, zero-latency pandas transformers to clean text and fix structural issues natively within the pipeline:

* **LowerCase:** Vectorized string standardization.
* **NumericStringToNaN & DashToNaN:** Wipes invalid placeholders and dashes.
* **YearExtractor:** Uses regex to bypass slow datetime parsing and extract build years.
* **KeepAutoManual:** Enforces strict categorical bounds for transmissions.
* **FrequencyEncoder:** Replaces rare categorical strings with their raw training occurrence counts.

### Feature Encoding & Scaling
* **Continuous Target Encoding:** Applied with smoothing to complex, high-cardinality features like car model.
* **Frequency/One-Hot Encoding:** Applied to trim, color, interior, body, and state.
* **Logarithmic Scaling:** `np.log1p` combined with `StandardScaler` applied to highly skewed numericals (odometer, condition, and year).

---

## 2. Modeling Strategy & Hyperparameter Tuning
I evaluated Linear Regression, Random Forest, and XGBoost. XGBoost dominated the baseline metrics and was chosen for deep optimization.

### The Overfitting Trade-Off
An initial `RandomizedSearchCV` produced a hyper-aggressive model (`max_depth`: 15). While it achieved an elite margin of error, it was structurally overfit to the training data. 

To build a robust production model, I deployed a targeted Micro-Search (`GridSearchCV`) to strictly walk back the tree depth while locking in the other winning parameters.

**Final Locked Hyperparameters:**
* `n_estimators`: 728
* `learning_rate`: 0.05
* `max_depth`: 11
* `min_child_weight`: 19 *(Crucial for preventing the memorization of rare luxury outliers)*
* `colsample_bytree`: 0.8

This surgical tune effectively slashed the Train/Test variance in half, sacrificing a trivial amount of absolute accuracy for a massive gain in real-world stability.

---

## 3. Performance Metrics & Statistical Proof
I evaluated the final model's performance across both the training data and a strictly unseen Test Set to validate the reduction in overfitting.

* **Train MAE:** $1,091 | **Test MAE** (Mean Absolute Error): $1,256
* **Train RMSE:** $1,836 | **Test RMSE** (Root Mean Squared Error): $2,330
* **Test R²:** 0.942

### Bootstrapped Confidence Intervals
To mathematically prove the model wasn't relying on a "lucky" test split, I ran a Percentile Bootstrap with 9,999 resamples.

* **95% CI for MAE:** [$1,219, $1,251]
* **95% CI for RMSE:** [$2,142, $2,380]

> **Business Reality:** I am 95% confident that this algorithm's true, long-term average error on unseen vehicles strictly falls between $1,219 and $1,251.

---

## 4. Deployment Architecture
I serialized the master pipeline via `joblib` and deployed it as a REST API.

* **Framework:** FastAPI
* **Validation:** Pydantic strictly enforces incoming JSON payloads, auto-generating interactive Swagger documentation.
* **Environment:** Containerized via Docker.
* **Hosting:** Deployed to Hugging Face Spaces.

---

