
# IT Ticket Routing Classification Model

This project is an automated machine learning pipeline designed to read incoming IT support tickets and route them to the correct department. It uses a custom-built scikit-learn architecture optimized for speed, reliability, and handling human error in the raw data.

## The Core Architecture

The text processing and modeling rely on a straightforward, high-efficiency pipeline:
* **Missing Data Imputation:** Empty tickets are replaced with a constant string.
* **Text Vectorization (TF-IDF):** Extracts the top 1500 n-grams (1-2 words), filtering out generic English stop words and extremely common/rare terms.
* **Classifier:** A Logistic Regression model using a One-vs-Rest (OVR) strategy to calculate distinct mathematical probabilities for each department.

## Critical Engineering Decisions

### 1. Dropping the "Miscellaneous" Class from Training
When analyzing the raw dataset, it became obvious that the "Miscellaneous" category was not a mathematical pattern. It was a dumping ground used by human agents who didn't want to find the correct dropdown category for complex tickets (like Oracle or timesheet issues). 

Training a model on human laziness destroys its ability to learn. I completely dropped the "Miscellaneous" class from the training set. This forced the model to learn the strict, healthy vocabulary patterns of the 7 actual IT departments. 

### 2. Choosing Logistic Regression over XGBoost
After building pipelines for both Logistic Regression and XGBoost, I ran rigorous hyperparameter tuning (`GridSearchCV` and `RandomizedSearchCV`) on both. They achieved the exact same average precision and F1 scores. 

I discarded XGBoost. There is zero value in maintaining a heavy, computationally expensive tree-based model in production when a simple, lightweight linear model accomplishes the exact same task.

### 3. Custom Probability Scaling for Minority Classes
The model initially struggled to accurately identify the "Administrative rights" class due to a lack of training examples. Instead of artificially generating fake data (SMOTE), I built a custom Python class: `ThresholdWrapper_and_ZeroVectorCatcher`. 

This wrapper dynamically scales the prediction threshold. By lowering the required confidence for "Administrative rights" to 25%, the model became far more aggressive at catching these specific tickets, successfully pushing the recall for this weak class to 0.85 without breaking the underlying math.

### 4. The Zero-Vector Catcher and Noise Floor
Because I removed "Miscellaneous" during training, I needed a safety net to handle actual garbage tickets (e.g., keyboard smashes or hopelessly ambiguous text) in the real world. The custom wrapper handles this in two phases before a final prediction is made:

* **The Zero-Vector Catcher:** If a ticket contains zero recognized IT words from the TF-IDF vocabulary, the wrapper intercepts it and forces the label to "Miscellaneous", completely bypassing the model's math.
* **The Noise Floor:** I mathematically calculated the model's absolute lowest correct prediction score (24%). The wrapper actively monitors every prediction; if the model's highest confidence fails to reach 0.24, it is flagged as a blind guess and forcefully routed to "Miscellaneous".

## Final Production Performance

The final model was evaluated on a completely unseen test set of 8,156 tickets. The combination of the Logistic Regression model and the custom fallback wrapper resulted in highly stable, production-ready metrics.

### High-Level Metrics
* **Accuracy:** 0.87
* **Macro F1-Score:** 0.86
* **Macro Precision:** 0.87
* **Macro Recall:** 0.86

### Department Breakdown

| Department | Precision | Recall | F1-Score | Support (Tickets) |
| :--- | :--- | :--- | :--- | :--- |
| **Purchase** | 0.97 | 0.87 | 0.92 | 493 |
| **Access** | 0.91 | 0.89 | 0.90 | 1425 |
| **Storage** | 0.94 | 0.84 | 0.89 | 555 |
| **HR Support** | 0.88 | 0.87 | 0.87 | 2183 |
| **Internal Project** | 0.90 | 0.84 | 0.87 | 424 |
| **Hardware** | 0.84 | 0.86 | 0.85 | 2724 |
| **Administrative rights** | 0.65 | 0.85 | 0.74 | 352 |
