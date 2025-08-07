import pandas as pd
import numpy as np
import pandas.api.types as ptypes
from .base_transformer import BaseTransformer


class DtypeOptimizer(BaseTransformer):
    """
    Optimizes pandas DataFrame dtypes to reduce memory usage.
    - Downcasts numeric types to the smallest possible subtype.
    - Converts object columns with low cardinality to 'category'.
    - Preserves existing categorical dtypes correctly.
    """

    def __init__(self, verbose=False):
        """
        Initialize the transformer.
        Args:
            verbose (bool): If True, enables detailed logging.
        """
        super().__init__(verbose=verbose)
        self.optimizations_ = {}
        # Define specific dtype conversions for XGBoost compatibility
        self.xgboost_dtype_conversions = {
            'BankruptcyHistory': 'float64',
            'PreviousLoanDefaults': 'category', 
            'PaymentHistory': 'int64',
            'UtilityBillsPaymentHistory': 'float64',
            'LoanPurpose': 'category'
        }

    def fit(self, X: pd.DataFrame, y=None):
        """
        Analyzes the DataFrame to determine the optimal dtype for each column.
        """
        super().fit(X, y)
        self._log_transformation("Fitting DtypeOptimizer...")

        for col in self.feature_names_in_:
            # Apply XGBoost-specific dtype conversions first
            if col in self.xgboost_dtype_conversions:
                self.optimizations_[col] = self.xgboost_dtype_conversions[col]
                self._log_transformation(f"XGBoost conversion: '{col}' -> {self.xgboost_dtype_conversions[col]}")
                continue
                
            dtype = X[col].dtype

            # CRITICAL: Preserve existing CategoricalDtype objects
            if isinstance(dtype, pd.CategoricalDtype):
                self.optimizations_[col] = dtype
                continue

            if dtype == 'object':
                num_unique = X[col].nunique()
                if num_unique / len(X) < 0.5:
                    self.optimizations_[col] = 'category'
                else:
                    self.optimizations_[col] = 'object' # No change
                continue

            # Handle Integer Columns
            if ptypes.is_integer_dtype(dtype):
                if X[col].hasnans:
                    # Use pandas nullable integer types for integers with NaNs
                    self.optimizations_[col] = pd.Int64Dtype.name
                else:
                    min_val, max_val = X[col].min(), X[col].max()
                    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                        self.optimizations_[col] = 'int8'
                    elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                        self.optimizations_[col] = 'int16'
                    elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                        self.optimizations_[col] = 'int32'
                    else:
                        self.optimizations_[col] = 'int64'
                continue

            # Handle Floating-Point Columns
            elif ptypes.is_float_dtype(dtype):
                # Downcast to float32 if no significant precision loss
                if np.allclose(X[col].dropna(), X[col].dropna().astype(np.float32), rtol=1e-5, atol=1e-5):
                    self.optimizations_[col] = 'float32'
                else:
                    self._log_transformation(f"Significant precision loss detected for '{col}'. Keeping float64.", 'warning')
                    self.optimizations_[col] = 'float64'
                continue
        
        self.feature_names_out_ = self.feature_names_in_ # DtypeOptimizer does not change column names
        self._log_transformation(f"DtypeOptimizer fitting complete. Optimizations: {self.optimizations_}")
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the determined dtype optimizations to the DataFrame.
        """
        initial_memory = X.memory_usage(deep=True).sum()
        self._log_transformation(f"Initial memory usage: {initial_memory / 1024**2:.2f} MB")

        for col, new_dtype in self.optimizations_.items():
            if col not in X.columns:
                self._log_transformation(f"Column '{col}' not found in transform data. Skipping.", 'warning')
                continue

            original_dtype = X[col].dtype
            if original_dtype == new_dtype:
                continue

            try:
                X[col] = X[col].astype(new_dtype)
                self._log_transformation(f"Optimized column '{col}': {original_dtype} -> {X[col].dtype}")
            except Exception as e:
                self._log_transformation(f"Could not convert '{col}' to {new_dtype}. Error: {e}", 'error')

        final_memory = X.memory_usage(deep=True).sum()
        if initial_memory > 0:
            saved = initial_memory - final_memory
            percent_saved = (saved / initial_memory) * 100
            self._log_transformation(f"Optimization complete. Memory saved: {saved / 1024**2:.2f} MB ({percent_saved:.2f}%)")

        return X