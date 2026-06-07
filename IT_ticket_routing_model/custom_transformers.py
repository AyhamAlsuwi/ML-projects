import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Flatten(BaseEstimator, TransformerMixin):
    """
    Efficiently flattens 2D array-like structures into 1D arrays.
    Strictly compliant with the scikit-learn API.
    """
    def fit(self, X, y=None):
        # Scikit-learn estimators should define an attribute trailing with an underscore 
        # during fit() to pass check_is_fitted() tests later if needed.
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        # np.ravel() creates a 1D view, maintaining O(1) latency.
        return np.ravel(X)

    def get_feature_names_out(self, input_features=None):
        """
        Required method for pipeline feature tracking.
        Prevents crashes when calling pipeline.get_feature_names_out().
        """
        # If the previous step (like an imputer) passes its feature names forward:
        if input_features is not None:
            # Since we flatten to a single text array, we return the first feature's name
            return np.array([input_features[0]], dtype=object)
        
        # Fallback generic name if no input features are provided
        return np.array(["flattened_text"], dtype=object)


from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def average_precision(average="macro"):
    """
    Creates a scikit-learn compatible scorer for multiclass Average Precision.
    Dynamically binarizes labels to prevent cross-validation crashes.
    """
    def multiclass_ap_scorer(estimator, X, y_true):
        # 1. Ensure the model actually outputs probabilities
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError("The estimator must have a 'predict_proba' method.")
            
        # 2. Get the probability predictions (required for PR curves)
        y_proba = estimator.predict_proba(X)
        
        # 3. Extract the exact classes the estimator learned during fit().
        # This guarantees column alignment between y_true and y_proba.
        classes = estimator.classes_
        
        # 4. Binarize true labels into One-vs-Rest indicator matrix (n_samples, n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # 5. Edge-case protection: If a specific CV fold drops to only 2 classes
        if len(classes) == 2:
            # label_binarize returns a 1D array for binary targets, 
            # and we only need the positive class probabilities from y_proba.
            return average_precision_score(y_true_bin, y_proba[:, 1])
            
        # 6. Compute and return the target multiclass metric
        return average_precision_score(y_true_bin, y_proba, average=average)
        
    return multiclass_ap_scorer



from sklearn.base import BaseEstimator, ClassifierMixin



class ThresholdWrapper_and_ZeroVectorCatcher(BaseEstimator, ClassifierMixin):
    """
    The complete production meta-model. 
    It handles Zero-Vector Catching, Probability Scaling, and Confidence Fallbacks.
    """
    def __init__(self, pipeline, custom_thresholds=None, fallback_label='Miscellaneous', min_confidence=None):
        self.pipeline = pipeline
        self.custom_thresholds = custom_thresholds
        self.fallback_label = fallback_label
        self.min_confidence = min_confidence

    def fit(self, X, y, **fit_params):
        self.pipeline.fit(X, y, **fit_params)
        
        # Extract classes from the final step of the pipeline
        final_step_name = list(self.pipeline.named_steps.keys())[-1]
        self.classes_ = self.pipeline.named_steps[final_step_name].classes_
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        # ---------------------------------------------------------
        # PHASE 1: THE ZERO-VECTOR CATCHER (Before math is applied)
        # ---------------------------------------------------------
        # Push text through everything EXCEPT the final classifier step [-1]
        sparse_matrix = self.pipeline[:-1].transform(X)
        recognized_word_counts = sparse_matrix.getnnz(axis=1)
        zero_vector_mask = (recognized_word_counts == 0)

        # ---------------------------------------------------------
        # PHASE 2: BASE PROBABILITIES & CUSTOM SCALING
        # ---------------------------------------------------------
        proba = self.predict_proba(X)
        max_raw_proba = np.max(proba, axis=1)
        
        model_classes = getattr(self, 'classes_', None)
        if model_classes is None:
            final_step = list(self.pipeline.named_steps.keys())[-1]
            model_classes = self.pipeline.named_steps[final_step].classes_
            
        thresholds = np.ones(len(model_classes))
        custom_dict = self.custom_thresholds or {}
        
        for class_name, threshold_val in custom_dict.items():
            if class_name not in model_classes:
                raise ValueError(f"Class '{class_name}' not found.")
            class_idx = np.where(model_classes == class_name)[0][0]
            thresholds[class_idx] = threshold_val
            
        scaled_proba = proba / thresholds
        best_indices = np.argmax(scaled_proba, axis=1)
        final_predictions = model_classes[best_indices].astype(object)
        
        # ---------------------------------------------------------
        # PHASE 3: THE FALLBACK OVERRIDES
        # ---------------------------------------------------------
        # 1. Apply the minimum confidence threshold to weak predictions
        if self.min_confidence is not None:
            final_predictions[max_raw_proba < self.min_confidence] = self.fallback_label
            
        # 2. OVERRIDE EVERYTHING for the Zero-Vectors. 
        # Even if the intercept gave Hardware a 0.45 score, this physically crushes it.
        final_predictions[zero_vector_mask] = self.fallback_label
            
        return final_predictions