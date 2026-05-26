import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin

class ShiftByOne(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X + 1
        
    def get_feature_names_out(self, input_features=None):
        # Simply pass the feature names through unchanged
        if input_features is None:
            return None
        return np.array(input_features)


from sklearn.metrics import average_precision_score

import numpy as np
from sklearn.metrics import average_precision_score

class average_precision_ovr:
    """
    A unified, zero-latency multiclass Average Precision scorer.
    Dynamically aligns y_true to match the model's internal classes, 
    making it natively compatible with both string and integer targets.
    """
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, estimator, X, y_true):
        # 1. Grab raw Softmax or Multi:softprob probabilities
        y_score = estimator.predict_proba(X)
        
        # 2. Dynamically extract the explicit order of classes the model trained on
        # This works for both string labels and integer labels
        model_classes = estimator.classes_
        
        # 3. Blazing fast mapping vectorization
        # Create a dynamic lookup map based on what the estimator expected
        cat_to_idx = {cat: idx for idx, cat in enumerate(model_classes)}
        
        # Map y_true directly to the correct matrix column index
        y_true_numeric = np.array([cat_to_idx[val] for val in y_true], dtype=np.int32)
        
        # 4. Direct C-level memory initialization for the one-hot matrix
        y_true_binarized = np.zeros_like(y_score, dtype=np.float32)
        y_true_binarized[np.arange(len(y_true_numeric)), y_true_numeric] = 1.0

        # 5. Compute the final metric score
        return average_precision_score(y_true_binarized, y_score, average=self.average)


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """
    A minimal, zero-latency wrapper responsible ONLY for executing 
    custom decision threshold overrides on multiclass probability matrices.
    """
    def __init__(self, pipeline, thresholds_dict=None):
        self.pipeline = pipeline
        self.thresholds_dict = thresholds_dict if thresholds_dict is not None else {}
        self._is_fitted = True  # Instantly bypass scikit-learn's fit validation

    @property
    def classes_(self):
        """Dynamically expose target classes directly from the underlying model."""
        if hasattr(self.pipeline, 'classes_'):
            return self.pipeline.classes_
        elif hasattr(self.pipeline, 'steps'):
            return self.pipeline.steps[-1][1].classes_
        raise AttributeError("Underlying pipeline does not expose 'classes_'.")

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        """
        Purely applies threshold rules using fast NumPy matrix vectorization.
        Automatically resolves string keys to integer matrix columns safely.
        """
        proba = self.predict_proba(X)
        classes_array = np.array(self.classes_)
        
        # 1. Establish standard argmax predictions as the base array
        predictions_idx = np.argmax(proba, axis=1)

        # 2. Apply custom threshold overrides safely
        for target_class, threshold in self.thresholds_dict.items():
            
            # SAFE LOOKUP: Resolve string labels ('B') or integer indices (1) to column positional index
            if target_class in classes_array:
                # If passed 'B', find where it is in ['A', 'B', 'C', 'D'] -> index 1
                target_idx = np.where(classes_array == target_class)[0][0]
            elif isinstance(target_class, int) and target_class < len(classes_array):
                # If passed 1, keep it as index 1
                target_idx = target_class
            else:
                 raise ValueError(f"Class key '{target_class}' from thresholds_dict not found in model classes {classes_array}.")

            # Isolate the resolved column probability matrix slice cleanly using an integer index
            target_proba = proba[:, target_idx]
            
            # Vectorized logical evaluation across the entire batch
            meets_threshold = target_proba >= threshold
            
            # Override standard argmax choices where thresholds are cleared
            predictions_idx[meets_threshold] = target_idx
            
        # 3. Return the mapped original labels matching what the model outputs
        return classes_array[predictions_idx]