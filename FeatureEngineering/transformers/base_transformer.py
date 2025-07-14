import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union, Optional
from pandas.api.types import is_numeric_dtype
from abc import ABC, abstractmethod

class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for all transformers in the feature engineering pipeline.
    Implements common functionality for validation, logging, and data type optimization.
    
    Parameters
    ----------
    verbose : bool, default=False
        If True, enable logging of transformation details.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _validate_input(self, X: pd.DataFrame, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Validate input DataFrame and ensure it meets requirements.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate
        required_columns : List[str], optional
            List of column names that must be present in the DataFrame
            
        Returns
        -------
        pd.DataFrame
            Validated copy of the input DataFrame
        
        Raises
        ------
        ValueError
            If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        if required_columns:
            missing_cols = set(required_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Required columns missing: {missing_cols}")
        
        return X.copy()

    def _log_transformation(self, message: str, level: str = "info") -> None:
        """
        Log transformation messages at specified level when verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            Message to log
        level : str, default="info"
            Logging level ('info', 'warning', or 'error')
        """
        if self.verbose:
            if level.lower() == "warning":
                self.logger.warning(message)
            elif level.lower() == "error":
                self.logger.error(message)
            else:
                self.logger.info(message)

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on the input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit the transformer
        y : ignored
            Not used, present for API consistency
            
        Returns
        -------
        self
            Return self for method chaining
        """
        X = self._validate_input(X)
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = self.feature_names_in_  # Default to input names
        self.is_fitted_ = True
        
        self._log_transformation(
            f"Successfully fitted {self.__class__.__name__} on {len(X)} rows"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Main transform method with built-in validation, error handling, and index alignment.

        This method wraps the `_transform` method, which must be implemented by subclasses.
        It ensures that the DataFrame's index is preserved and handles exceptions gracefully.
        """
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} is not fitted yet.")

        X_validated = self._validate_input(X)
        original_index = X_validated.index

        try:
            X_transformed = self._transform(X_validated)
        except Exception as e:
            self.logger.error(f"Error during transformation in {self.__class__.__name__}: {e}")
            # Return a DataFrame with the original index and no columns to signal failure
            return pd.DataFrame(index=original_index)

        # --- Index Alignment Validation ---
        if not X_transformed.index.equals(original_index):
            self.logger.warning(
                f"Index of transformed data in {self.__class__.__name__} does not match original index. "
                f"Realigning to original index. This may indicate an issue in the transformer."
            )
            X_transformed = X_transformed.reindex(original_index)

        self.feature_names_out_ = list(X_transformed.columns)
        self._log_transformation(
            f"Successfully transformed data. Output shape: {X_transformed.shape}"
        )
        return X_transformed

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The core transformation logic to be implemented by subclasses.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        pass

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : ignored
            Not used, present for API consistency
            
        Returns
        -------
        List[str]
            List of output feature names
        """
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} is not fitted yet.")
        return self.feature_names_out_

    def _optimize_numeric_dtype(self, series: pd.Series) -> pd.Series:
        """
        Optimize numeric data type based on value range.
        
        Parameters
        ----------
        series : pd.Series
            Input series to optimize
            
        Returns
        -------
        pd.Series
            Series with optimized dtype
        """
        if not is_numeric_dtype(series):
            return series
            
        min_val = series.min()
        max_val = series.max()
        
        # For integers
        if series.dtype.kind in ['i', 'u'] or series.apply(float.is_integer).all():
            if min_val >= 0:
                if max_val <= 255:
                    return series.astype(np.uint8)
                elif max_val <= 65535:
                    return series.astype(np.uint16)
                elif max_val <= 4294967295:
                    return series.astype(np.uint32)
                return series.astype(np.uint64)
            else:
                if min_val >= -128 and max_val <= 127:
                    return series.astype(np.int8)
                elif min_val >= -32768 and max_val <= 32767:
                    return series.astype(np.int16)
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return series.astype(np.int32)
                return series.astype(np.int64)
        
        # For floats
        if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            return series.astype(np.float32)
        return series.astype(np.float64)

    def _optimize_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize dtypes for all columns in DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to optimize
            
        Returns
        -------
        pd.DataFrame
            DataFrame with optimized dtypes
        """
        df = df.copy()
        
        for col in df.columns:
            # Optimize numeric columns
            if is_numeric_dtype(df[col]):
                df[col] = self._optimize_numeric_dtype(df[col])
            # Optimize categorical/object columns
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
        
        self._log_transformation(
            f"Optimized data types. Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB"
        )
        return df