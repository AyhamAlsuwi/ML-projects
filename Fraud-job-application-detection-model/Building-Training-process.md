# Fraudulent Job Posting Detection: End-to-End Retrospective

## 1. Project Overview
This project focused on building a machine learning pipeline to detect fraudulent job postings. The core dataset presented a severe class imbalance problem: 95% of the data represented legitimate jobs, while only 5% were fraudulent. 

The primary objective was to architect a robust data engineering and classification pipeline capable of handling messy, unstructured text and metadata, ultimately resulting in a deployed inference API.

## 2. The Data Engineering Architecture
Handling raw, unstructured job postings required a dual-pronged approach to feature engineering. Relying exclusively on text or exclusively on metadata would fail to capture the full scope of a fraudulent posting.

### 2.1 Lexical Analysis (TF-IDF)
Fraudulent jobs often use specific, urgent, or manipulative language ("Cash paid weekly", "Work from home data entry"). To capture the semantic meaning of the job descriptions, company profiles, and requirements, I implemented **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. This allowed the model to statistically weigh the importance of specific vocabulary associated with scams.

### 2.2 Structural Metadata & Custom Transformers
Text alone was insufficient because fraudulent postings consistently lacked verifiable corporate metadata. To capture this, I engineered custom Scikit-Learn transformers to work alongside the TF-IDF vectors:
* **PresenceEncoder**: Converted raw text fields into binary presence indicators (1 for present, 0 for missing). The *absence* of a company profile proved to be a massive mathematical signal.
* **HighSalaryRange & SalaryPresence**: Parsed unstructured salary text to detect abnormal compensation ranges often used as bait.
* **Flatten**: Managed array dimensions to ensure dense engineered features could concatenate cleanly with sparse TF-IDF matrices.

These data streams were merged using a `ColumnTransformer` to feed a unified matrix to the XGBoost engine.

## 3. Key Roadblocks & Resolutions
The development of this model presented several intense data science and engineering challenges before the pipeline was stable.

| Struggle / Roadblock | Root Cause | How I Resolved It |
| :--- | :--- | :--- |
| **The Accuracy Illusion** | The 95/5 class imbalance meant the model could achieve 95% accuracy simply by predicting "Legitimate" for every single row, catching zero actual fraud. | Discarded standard accuracy. Rewrote the evaluation framework to optimize exclusively for the **Precision-Recall (PR) Curve** and F1-score. |
| **Dimensionality Collisions** | TF-IDF outputs sparse 2D matrices, while custom metadata transformers output 1D arrays or dense matrices. Scikit-Learn's `ColumnTransformer` crashed when attempting to merge them. | Engineered the custom `Flatten` transformer class to intercept and reshape arrays dynamically, ensuring perfect dimensional alignment prior to concatenation. |
| **The Missing Data Mirage** | The model trained perfectly on `NaN` (null) values, but during real-world testing, missing text inputs were often evaluated as empty strings (`""`). Pandas saw `""` as valid text, completely blinding the `PresenceEncoder`. | Wrote a regex preprocessing step to aggressively intercept the data matrix and convert all empty or whitespace-only strings into true Numpy `NaN` values before the pipeline executed. |

## 4. Model Evaluation & Business Logic
Instead of relying on a default binary `0.5` threshold, I analyzed the PR Curve to map the model's probabilities to distinct business actions. This translated raw math into actionable Trust & Safety protocols:

| Probability Threshold | Business Action | Precision | Recall | Application |
| :--- | :--- | :--- | :--- | :--- |
| **>= 0.80** | Auto-Ban (CRITICAL) | 98.1% | 60.2% | Instant removal; extremely low false positive rate. |
| **>= 0.40** | Human Review (HIGH) | 88.2% | 74.3% | The mathematical F1 sweet spot. Sent to a moderation queue. |
| **>= 0.17** | Initial Filter (ELEVATED) | 68.9% | 83.0% | High recall net. Flagged for secondary scanning. |
| **< 0.17** | Publish Job (LOW) | N/A | N/A | Cleared to be published on the platform. |

## 5. Personal Growth & Key Takeaways

### Feature Engineering > Algorithm Choice
I learned that an advanced algorithm like XGBoost cannot compensate for poor data representation. The defining factor in this project's success wasn't the hyperparameter tuning of the XGBoost trees; it was the custom `PresenceEncoder` and TF-IDF integration. Structuring the problem correctly proved far more valuable than the algorithm itself.

### Translating Math to Business
A raw probability score of `0.65` is useless to a product manager or a backend system. I learned how to bridge the gap between data science and product by translating the model's PR curve into deterministic severity tiers (Auto-Ban, Review, Filter). It taught me that a model's output must dictate a clear business action.

### Granular Pipeline Control
Moving away from off-the-shelf Scikit-Learn imputers and writing my own Transformer classes fundamentally changed how I view ML pipelines. Forcing myself to handle the `fit` and `transform` logic manually—and resolving the array dimensionality crashes that followed—gave me a much deeper understanding of how data physically moves through a machine learning architecture.
