import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class LowerCase(BaseEstimator, TransformerMixin):
    """
    High-performance Scikit-Learn transformer for lowercase conversion and whitespace stripping.
    Optimized for low latency by using vectorized pandas operations 
    and avoiding explicit loops and 'nan' string replacements.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Captures metadata and ensures input is compatible.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Vectorized transformation.
        """
        check_is_fitted(self)
        
        # Avoid creating a deep copy if not necessary; 
        # however, we must return a modified version.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            # Only attempt lowercase and strip on object/string columns to reduce latency
            if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
                # Chain .str.lower() and .str.strip() for zero-latency vectorized execution
                X[col] = X[col].str.lower().str.strip()
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)




class NumericStringToNaN(BaseEstimator, TransformerMixin):
    """
    High-performance transformer that converts both fully numeric strings 
    and actual numeric types (int/float) to NaN.
    Optimized for low latency using vectorized pandas operations.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Validates input and captures metadata.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Vectorized transformation targeting both strings and numeric types.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            col_data = X[col]
            
            # Scenario 1: Actual numeric types (int, float)
            if pd.api.types.is_numeric_dtype(col_data):
                X[col] = np.nan
                continue # Move to next column since this one is now entirely NaN
            
            # Scenario 2: String/Object columns that might contain numeric strings
            if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
                # .str.isnumeric() returns True/False/NaN.
                # Comparing exactly to True avoids the fillna() downcasting warning 
                # because NaN == True evaluates to False natively, resulting in a clean boolean mask.
                is_numeric_mask = col_data.str.isnumeric() == True
                
                # Apply vectorized replacement for numeric strings
                X.loc[is_numeric_mask, col] = np.nan
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)





class DashToNaN(BaseEstimator, TransformerMixin):
    """
    High-performance transformer that converts all common dash characters 
    ('-', '–', '—') into NaN. Optimized for low latency.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Validates input and captures metadata.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Vectorized transformation targeting multiple dash types.
        Uses infer_objects to satisfy FutureWarnings.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        # List of all common dash variants: hyphen, en-dash, em-dash
        dashes = ['-', '–', '—']

        # Vectorized replacement across all columns in the subset
        X[self.feature_names_in_] = X[self.feature_names_in_].replace(dashes, np.nan)
        
        # Prevent FutureWarnings by explicitly allowing pandas to re-infer dtypes
        X = X.infer_objects(copy=False)
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)






class YearExtractor(BaseEstimator, TransformerMixin):
    """
    High-performance transformer to extract a year from a verbose date string.
    Bypasses slow datetime parsing in favor of lightning-fast regex.
    Converts to Pandas 'Int64' to allow integers and NaNs to coexist.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Validates input and captures metadata.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Extracts the year using vectorized regex.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            # 1. Convert to string to avoid errors on pure NaNs
            # 2. Extract exactly 4 digits surrounded by spaces (e.g., " 2015 ")
            # 3. expand=False ensures we get a Series back, not a DataFrame
            years = X[col].astype(str).str.extract(r' (\d{4}) ', expand=False)
            
            # 4. Convert to Pandas Nullable Integer ('Int64')
            # Standard 'int' crashes on NaN. 'Int64' handles it perfectly.
            X[col] = years.astype('Int64')
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)


class KeepAutoManual(BaseEstimator, TransformerMixin):
    """
    High-performance transformer that enforces strict categorical limits.
    Any value that is not 'automatic' or 'manual' (case-insensitive, ignoring spaces) 
    is instantly converted to NaN.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Validates input and captures metadata.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Applies vectorized masking to wipe invalid categories.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            # 1. Normalize the data: cast to string and strip whitespace.
            # Assumes lowercasing is already handled upstream.
            normalized_col = X[col].astype(str).str.strip()
            
            # 2. Create a boolean mask of VALID rows
            is_valid_mask = normalized_col.isin(['automatic', 'manual'])
            
            # 3. Invert the mask (~) to target INVALID rows and set them to NaN
            X.loc[~is_valid_mask, col] = np.nan
            
        # 4. Explicitly satisfy FutureWarnings regarding downcasting
        X = X.infer_objects(copy=False)
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)




class LongStringToNaN(BaseEstimator, TransformerMixin):
    """
    High-performance transformer that converts any string longer than 5 characters to NaN.
    Optimized for zero latency using vectorized pandas string length evaluation.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Validates input and captures metadata.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """
        Applies vectorized masking to wipe strings longer than 5 characters.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            col_data = X[col]
            
            # Type guard: only execute on object/string columns. 
            # If applied to a pure numeric column, it ignores it instead of crashing.
            if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
                
                # 1. .str.len() gets the length. It ignores existing NaNs (keeps them NaN).
                # 2. > 5 creates a True/False/NaN mask.
                # 3. .fillna(False) prevents ValueError during indexing.
                # 4. .astype(bool) prevents the pandas downcasting FutureWarning.
                too_long_mask = (col_data.str.len() > 10).fillna(False).astype(bool)
                
                # Vectorized wipe
                X.loc[too_long_mask, col] = np.nan
                
        # Satisfy FutureWarnings
        X = X.infer_objects(copy=False)
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)




class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    High-performance Frequency/Count Encoder for categorical features.
    Replaces categories with their raw occurrence counts from the training set.
    Unseen categories during transform() are safely assigned a count of 0.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.frequency_maps_ = {}

    def fit(self, X, y=None):
        """
        Calculates and memorizes the absolute count of each category in the training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Build a dictionary of raw counts for every column
        for col in self.feature_names_in_:
            # normalize=False gives raw counts instead of proportions.
            # dropna=False ensures we also encode the frequency of missing data!
            self.frequency_maps_[col] = X[col].value_counts(normalize=False, dropna=False).to_dict()
            
        return self

    def transform(self, X):
        """
        Vectorized mapping of memorized counts to the new data.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        for col in self.feature_names_in_:
            # 1. Map the string to its memorized raw count.
            # 2. .fillna(0) is the lifesaver: if X_test has a category that wasn't 
            #    in X_train, it maps to NaN. We immediately fill it with 0.
            X[col] = X[col].map(self.frequency_maps_[col]).fillna(0)
            
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for pipeline compatibility.
        """
        check_is_fitted(self)
        return np.array(self.feature_names_in_)