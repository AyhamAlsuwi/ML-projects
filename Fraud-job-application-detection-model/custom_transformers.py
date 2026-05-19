import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin , ClassifierMixin
from sklearn.utils.validation import check_is_fitted



class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, threshold=0.50):
        self.pipeline = pipeline
        self.threshold = threshold
        # Inherit the classes from the underlying model so sklearn doesn't complain
        self.classes_ = pipeline.classes_ 

    def fit(self, X, y=None):
        # We don't need to fit anything, the pipeline is already trained
        return self

    def predict(self, X):
        # 1. Get the raw probabilities for the Fraud class (Class 1)
        # 2. Apply the custom threshold using hyper-fast NumPy vectorization
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        # Pass-through in case the API still wants to see the raw scores
        return self.pipeline.predict_proba(X)








class Flatten(BaseEstimator, TransformerMixin):
    """
    A low-latency Scikit-Learn transformer that flattens 2D arrays 
    into 1D arrays. Uses zero-copy views and conditional type-casting
    to minimize memory allocation overhead in production.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # We only need the shape and names for metadata, 
        # no heavy data transformations in fit.
        X_np = np.asarray(X)
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1

        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        elif hasattr(X, 'name') and X.name is not None:
            self.feature_names_in_ = np.array([X.name], dtype=object)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        
        # 1. Zero-Copy Flatten
        # np.ravel() is highly optimized to return a memory 'view' 
        # instead of a copy whenever mathematically possible.
        X_flat = np.ravel(X)

        # 2. The Latency Firewall (Conditional Casting)
        # 'O' = Object (Pandas default for text)
        # 'U', 'S' = Unicode/String (NumPy default for text)
        # We ONLY trigger the expensive .astype(str) if a stray float/int got through.
        if X_flat.dtype.kind not in ('O', 'U', 'S'):
            return X_flat.astype(str)

        return X_flat

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        
        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
            
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)




class PresenceEncoder(BaseEstimator, TransformerMixin):
    """
    Converts any feature into a binary flag: 
    1 if data is present (not null/empty), 0 if null/empty.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_np = np.asarray(X)
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        
        # Convert to DataFrame if it's not already, to use .notnull() 
        # which is more robust across mixed types than numpy logic
        if not hasattr(X, 'notnull'):
            X = pd.DataFrame(X)
            
        # 1. Check for nulls
        # 2. Convert boolean to int (True -> 1, False -> 0)
        return X.notnull().astype(int).values

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'is_fitted_')
        if input_features is not None:
            return np.array([f"{f}_has_value" for f in input_features], dtype=object)
        
        if hasattr(self, 'feature_names_in_'):
            return np.array([f"{f}_has_value" for f in self.feature_names_in_], dtype=object)
            
        return np.array([f"x{i}_has_value" for i in range(self.n_features_in_)], dtype=object)



class SalaryPresence(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to detect the presence of salary-related 
    text within job descriptions.
    
    Returns:
        np.ndarray: A 2D array of shape (n_samples, 1) containing 1 (Salary Found) 
                    or 0 (No Salary Found).
    """
    
    def __init__(self):
        # ---------------------------------------------------------------------
        # REGEX ARCHITECTURE (Using re.VERBOSE for readability)
        # ---------------------------------------------------------------------
        # We pre-compile this once during initialization to guarantee zero-latency 
        # inference during the .transform() phase.
        
        salary_pattern = r'''(?x)       # Enable VERBOSE mode (allows multiline and comments)
            (?i)                        # Make the entire search CASE-INSENSITIVE

            (?:                         # Start of main non-capturing group
            
                # -------------------------------------------------------------
                # TYPE 1: Currency Symbol Formats
                # Matches: "$100", "$ 50000", "€15"
                # -------------------------------------------------------------
                [\$\£\€]                # Must start with a currency symbol
                \s* # Optional whitespace
                \d+                     # One or more digits
                
                |                       # OR
                
                # -------------------------------------------------------------
                # TYPE 2: Rate-Based Formats (Hourly, Yearly, Monthly)
                # Matches: "100/hour", "50k per yr", "15.50 / hr"
                # -------------------------------------------------------------
                \d+[.,]?\d* # Base number (handles decimals like 15.50)
                \s*[kK]?                # Optional 'k' or 'K' multiplier (e.g., 50k)
                \s*(?:/|per)\s* # The separator: either '/' or 'per'
                (?:hr|hour|yr|year|mo|month) # The time period indicator
                
                |                       # OR
                
                # -------------------------------------------------------------
                # TYPE 3: Range & Explicit Word Formats
                # Matches: "15 - 20 dollars", "100k usd", "50 dollar"
                # -------------------------------------------------------------
                \d+                     # Starting number
                (?:\s*-\s*\d+)?         # Optional range suffix (e.g., " - 20")
                \s*[kK]?                # Optional 'k' or 'K' multiplier
                \s*(?:dollars?|usd)     # Explicit currency word (dollar, dollars, usd)
                
            )                           # End of main non-capturing group
        '''
        
        # Compile into C-backend state machine
        self.regex = re.compile(salary_pattern)
        
    def fit(self, X, y=None):
        """ Stateless transformer: Nothing to learn from the data. """
        return self
        
    def transform(self, X):
        """
        Executes the pre-compiled regex across the feature array.
        Includes built-in safety nets for Null values and Pandas/Numpy conversions.
        """
        # 1. Standardize input to a 1D Pandas Series for vectorized string ops
        if isinstance(X, np.ndarray):
            X_series = pd.Series(X.ravel())
        elif isinstance(X, pd.DataFrame):
            X_series = X.iloc[:, 0] 
        else:
            X_series = pd.Series(X)
            
        # 2. Safety Net: Kill nulls immediately to prevent pipeline crashes
        X_clean = X_series.fillna('')
        
        # 3. Vectorized execution: Check for pattern, convert True/False to 1/0
        salary_presence = X_clean.str.contains(self.regex).astype(int)
        
        # 4. Scikit-learn formatting: Reshape to 2D array [[1], [0], [1]]
        return salary_presence.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """
        Mandatory method for ColumnTransformer compatibility.
        Declares the exact name of the engineered feature for downstream analysis.
        """
        return np.array(['has_salary'])



class HighSalaryRange(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to detect suspiciously high numbers 
    in a raw "MIN - MAX" format (e.g., "50000 - 150000").
    
    Logic: Flags any range where either the MIN or MAX is >= 100,000 (6+ digits).
    Returns: 2D array of 1 (Suspicious) or 0 (Normal/NaN).
    """
    
    def __init__(self):
        # ---------------------------------------------------------------------
        # REGEX MATH ARCHITECTURE (re.VERBOSE)
        # ---------------------------------------------------------------------
        # We don't convert strings to floats. We just count the digits.
        # \d{6,} mathematically guarantees the number is 100,000 or higher.
        
        range_pattern = r'''(?x)
            (?:                         # Start of group
            
                # -------------------------------------------------------------
                # SCENARIO 1: The Max is absurdly high (The usual scam)
                # Matches: "50000 - 150000", "0 - 1000000"
                # -------------------------------------------------------------
                \b\d+                   # The MIN number (any length)
                \s*-\s* # The hyphen separator (with optional spaces)
                \b\d{6,}\b              # The MAX number MUST be 6 or more digits (100k+)
                
                |                       # OR
                
                # -------------------------------------------------------------
                # SCENARIO 2: The Min is absurdly high
                # Matches: "120000 - 150000", "200000-300000"
                # -------------------------------------------------------------
                \b\d{6,}\b              # The MIN number MUST be 6 or more digits (100k+)
                \s*-\s* # The hyphen separator
                \b\d+                   # The MAX number (any length)
                
            )                           # End of group
        '''
        
        # Compile into C-backend for zero latency
        self.regex = re.compile(range_pattern)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 1. Standardize format for Scikit-Learn compatibility
        if isinstance(X, np.ndarray):
            X_series = pd.Series(X.ravel())
        elif isinstance(X, pd.DataFrame):
            X_series = X.iloc[:, 0] 
        else:
            X_series = pd.Series(X)
            
        # 2. NAN HANDLING: Fill NaN with empty string. 
        # Fails the regex natively, resulting in a perfect '0'.
        X_clean = X_series.fillna('')
        
        # 3. Vectorized Execution: Check for the 100k+ pattern
        high_range_flag = X_clean.str.contains(self.regex).astype(int)
        
        # 4. Return as 2D array
        return high_range_flag.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        # Explicitly declare the engineered column name
        return np.array(['suspicious_salary_range'])





