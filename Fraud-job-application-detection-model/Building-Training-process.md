# Fraudulent Job Posting Detection: Project Retrospective

## 1. Project Objective
The objective was to build a machine learning pipeline capable of identifying fraudulent job postings. The core challenge was severe class imbalance: approximately 95% of the dataset consisted of legitimate jobs, while only 5% were fraudulent. 

## 2. Methodology & Implementation

### 2.1. Feature Engineering & Custom Transformers
Instead of relying strictly on raw text processing, structural metadata was extracted. Fraudulent postings consistently lacked verifiable corporate information. 
Custom Scikit-Learn transformers were developed to extract these signals without causing memory bloat:
* `PresenceEncoder`: Converted text fields (like company profiles) into binary indicators (present vs. missing).
* `SalaryPresence` & `HighSalaryRange`: Extracted numerical bounds from unstructured salary strings to flag abnormal compensation ranges.
* `Flatten`: Handled array dimensionality issues between text vectors and categorical pipelines.

### 2.2. Preprocessing & Data Leakage Prevention
A `ColumnTransformer` was used to process text, categorical, and numerical features independently before combining them into a sparse matrix.
A critical data leakage issue was identified and fixed regarding missing values. Web payloads default to empty strings (`""`), which Pandas evaluates as valid text. This blinded the `PresenceEncoder` and `SimpleImputer`. The pipeline was modified to aggressively use regex to convert whitespace and empty strings into true `NaN` values prior to inference.

### 2.3. Model Selection & Training
An `XGBoostClassifier` was selected to handle the sparse matrix and non-linear relationships. 
Standard accuracy was discarded as an evaluation metric. In a 95/5 imbalanced dataset, a model predicting "legitimate" 100% of the time achieves 95% accuracy but is entirely useless. Optimization and evaluation were shifted entirely to the Precision-Recall (PR) curve.

### 2.4. Business Threshold Optimization (Test Set Results)
The default `0.5` probability threshold was discarded. The PR curve was used to map probabilities to specific business actions based on risk tolerance. 

| Threshold | Action Tier | Precision | Recall | Business Application |
| :--- | :--- | :--- | :--- | :--- |
| **>= 0.80** | Auto-Ban (Critical) | 98.1% | 60.2% | High confidence. Instant removal with minimal risk to legitimate users. |
| **>= 0.40** | Human Review (High) | 88.2% | 74.3% | The mathematical F1 sweet spot. Sent to a moderation queue. |
| **>= 0.17** | Initial Filter (Elevated) | 68.9% | 83.0% | High recall net. Flagged for secondary scanning. |
| **< 0.17** | Publish (Low) | N/A | N/A | Clean data. Allowed onto the platform. |

### 2.5. Deployment Architecture
The final pipeline was containerized and deployed as a REST API using FastAPI and Uvicorn on Hugging Face Spaces (Docker). 
* **Pydantic:** Used to enforce strict input schema validation.
* **Diagnostic Output:** The endpoint was engineered to return the raw probability, the assigned risk level, the recommended business action, and a boolean diagnostic breakdown of which thresholds were triggered.

## 3. Key Technical Takeaways

1.  **Accuracy is a liability in imbalanced data.** Optimizing for the F1 score and analyzing the Precision-Recall curve is the only mathematically sound way to evaluate fraud detection models.
2.  **Web data behaves differently than CSV data.** Models trained on static CSVs with `NaN` values will fail in production if they encounter empty strings (`""`) from JSON payloads. Strict input sanitization at the API layer is mandatory.
3.  **Probabilities must be translated into deterministic actions.** Engineering teams cannot build application logic around raw probability scores. Models must output discrete, actionable recommendations based on defined thresholds.
4.  **Serialization of custom classes is brittle.** Using `joblib` to save a pipeline containing custom Scikit-Learn transformers requires exact version matching (`requirements.txt`) and identical module pathing (`__init__.py`) between the local training environment and the production server. Mismatches result in fatal deployment errors.
