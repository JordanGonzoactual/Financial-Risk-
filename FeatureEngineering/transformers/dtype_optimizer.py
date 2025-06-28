import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for optimizing pandas DataFrame dtypes to reduce memory usage.
    This transformer is designed to work within a scikit-learn pipeline.

    It optimizes numeric and categorical columns.
    - Integers are downcast to the smallest possible subtype.
    - Floats are downcast from float64 to float32 where possible without significant precision loss.
    - Object columns with low cardinality are converted to 'category' dtype.
    - Special handling for financial, polynomial, and encoded categorical features.
    """

    def __init__(self):
        self.optimizations_ = {}
        self.original_dtypes_ = {}

    def fit(self, X, y=None):
        """
        Analyzes the input DataFrame to determine the optimal dtype for each column.

        Args:
            X (pd.DataFrame): The input data to analyze.
            y (None): Ignored. This parameter is included for sklearn compatibility.

        Returns:
            self: The fitted transformer instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self.original_dtypes_ = X.dtypes.to_dict()
        self.optimizations_ = {}

        for col in X.columns:
            dtype = X[col].dtype

            # --- Categorical Feature Optimization ---
            # Preserve categorical dtype for specified features or those already encoded.
            if dtype.name == 'category' or col in ['EducationLevel', 'life_stage']:
                self.optimizations_[col] = 'category'
                continue
            
            if dtype == 'object':
                # Convert object columns with low cardinality to category
                num_unique_values = X[col].nunique()
                num_total_values = len(X[col])
                if num_total_values > 0 and num_unique_values / num_total_values < 0.5:
                    self.optimizations_[col] = 'category'
                else:
                    self.optimizations_[col] = X[col].dtype # keep as is
                continue

            # --- Numeric Feature Optimization ---
            if np.issubdtype(dtype, np.integer):
                min_val, max_val = X[col].min(), X[col].max()
                if pd.isna(min_val) or pd.isna(max_val):
                    self.optimizations_[col] = X[col].dtype # keep as is if column is all NaN
                    continue
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    self.optimizations_[col] = 'int8'
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    self.optimizations_[col] = 'int16'
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    self.optimizations_[col] = 'int32'
                else:
                    self.optimizations_[col] = 'int64'
                continue

            if np.issubdtype(dtype, np.floating):
                # --- Polynomial and Financial Feature Optimization ---
                # For polynomial features, be cautious with precision.
                # For others, float32 is often sufficient.
                is_poly = 'poly' in col or '^' in col or '*' in col
                
                # Check for potential overflow before converting
                if np.any(np.isinf(X[col].astype(np.float32))):
                     logging.warning(f"Column '{col}' contains values that would overflow float32. Keeping float64.")
                     self.optimizations_[col] = 'float64'
                     continue

                # Smart detection for precision loss
                if is_poly:
                    # More stringent check for polynomial features
                    if np.allclose(X[col].dropna(), X[col].dropna().astype(np.float32), rtol=1e-5, atol=1e-5):
                        self.optimizations_[col] = 'float32'
                    else:
                        logging.warning(f"Significant precision loss detected for polynomial feature '{col}'. Keeping float64.")
                        self.optimizations_[col] = 'float64'
                else:
                    # Standard conversion for other floats
                    self.optimizations_[col] = 'float32'
                continue
        
        logging.info("DtypeOptimizer fitting complete. Determined optimal dtypes.")
        return self

    def transform(self, X, y=None):
        """
        Applies the determined dtype optimizations to the input DataFrame.

        Args:
            X (pd.DataFrame): The input data to transform.
            y (None): Ignored. This parameter is included for sklearn compatibility.

        Returns:
            pd.DataFrame: The transformed DataFrame with optimized dtypes.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        X_transformed = X.copy()
        
        initial_memory = X_transformed.memory_usage(deep=True).sum()
        logging.info(f"Initial memory usage: {initial_memory / 1024**2:.2f} MB")

        for col, new_dtype in self.optimizations_.items():
            if col not in X_transformed.columns:
                logging.warning(f"Column '{col}' from fit stage not found in transform stage. Skipping.")
                continue

            original_dtype = X_transformed[col].dtype
            if str(original_dtype) == new_dtype:
                continue

            try:
                # --- Validation before conversion ---
                original_min, original_max = None, None
                if np.issubdtype(original_dtype, np.number) and X_transformed[col].notna().any():
                    original_min = X_transformed[col].min()
                    original_max = X_transformed[col].max()

                # Apply optimization
                if new_dtype == 'category' or isinstance(new_dtype, pd.CategoricalDtype):
                    X_transformed[col] = X_transformed[col].astype('category')
                else:
                    X_transformed[col] = X_transformed[col].astype(new_dtype)

                # --- Validation after conversion ---
                if np.issubdtype(X_transformed[col].dtype, np.number) and original_min is not None:
                    new_min, new_max = X_transformed[col].min(), X_transformed[col].max()
                    # Use a tolerance for float comparisons
                    if not (np.allclose(original_min, new_min) and np.allclose(original_max, new_max)):
                        logging.error(
                            f"Data range changed for column '{col}' after casting to {new_dtype}. "
                            f"Original range: ({original_min}, {original_max}), "
                            f"New range: ({new_min}, {new_max}). Reverting."
                        )
                        # Fallback strategy
                        X_transformed[col] = X[col]
                    else:
                         logging.info(f"Optimized column '{col}': {original_dtype} -> {new_dtype}")
                else:
                    logging.info(f"Optimized column '{col}': {original_dtype} -> {new_dtype}")

            except Exception as e:
                logging.error(f"Could not convert column '{col}' to {new_dtype}. Error: {e}. Keeping original dtype.")
                # Fallback strategy
                X_transformed[col] = X[col]
        
        final_memory = X_transformed.memory_usage(deep=True).sum()
        logging.info(f"Final memory usage: {final_memory / 1024**2:.2f} MB")

        if initial_memory > 0:
            memory_saved = initial_memory - final_memory
            percentage_saved = (memory_saved / initial_memory) * 100
            logging.info(f"Memory saved: {memory_saved / 1024**2:.2f} MB ({percentage_saved:.2f}%)")
        
        return X_transformed