import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts year, month, and day components from a date column
    and removes the original date column.
    """
    
    def __init__(self, date_column='ApplicationDate', verbose=False):
        """
        Initialize the DateTransformer.
        
        Parameters:
        -----------
        date_column : str, default='ApplicationDate'
            Name of the date column to transform
        verbose : bool, default=False
            Whether to log transformation details
        """
        self.date_column = date_column
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        """
        Validate that the date column exists and can be parsed as datetime.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame containing the date column
        y : ignored
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Check if the date column exists in the DataFrame
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in input DataFrame")
        
        # Try parsing the date column to validate it can be converted to datetime
        try:
            pd.to_datetime(X[self.date_column], errors='raise')
        except Exception as e:
            raise ValueError(f"Failed to parse '{self.date_column}' as datetime: {str(e)}")
        
        # Store input feature names
        self.input_features_ = X.columns.tolist()
        
        # Generate output feature names
        # Remove the date column and add year, month, day component columns
        date_col_lower = self.date_column.lower()
        self.output_features_ = [col for col in self.input_features_ if col != self.date_column]
        self.output_features_.extend([
            f"{date_col_lower}_year",
            f"{date_col_lower}_month",
            f"{date_col_lower}_day"
        ])
        
        if self.verbose:
            self.logger.info(f"DateTransformer will extract components from '{self.date_column}' "
                            f"and create 3 new features")
        
        return self
    
    def transform(self, X):
        """
        Transform the date column by extracting year, month, and day components
        and removing the original column.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame containing the date column
        
        Returns:
        --------
        X_transformed : pandas.DataFrame
            Transformed DataFrame with date components and without original date column
        """
        # Check if fit has been called
        check_is_fitted(self, ["input_features_", "output_features_"])
        
        # Make a copy of the input DataFrame to avoid modifying the original
        X_transformed = X.copy()
        
        try:
            # Parse dates
            dates = pd.to_datetime(X_transformed[self.date_column], errors='coerce')
            
            # Extract date components with optimized data types
            date_col_lower = self.date_column.lower()
            X_transformed[f"{date_col_lower}_year"] = dates.dt.year.astype('int16')
            X_transformed[f"{date_col_lower}_month"] = dates.dt.month.astype('int8')
            X_transformed[f"{date_col_lower}_day"] = dates.dt.day.astype('int8')
            
            if self.verbose:
                self.logger.info(f"Created date component columns from '{self.date_column}'")
            
            # Remove the original date column
            X_transformed = X_transformed.drop(columns=[self.date_column])
            
            if self.verbose:
                self.logger.info(f"Removed original date column '{self.date_column}'")
                
        except Exception as e:
            raise RuntimeError(f"Error during date transformation: {str(e)}")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features (ignored)
        
        Returns:
        --------
        feature_names_out : ndarray of str objects
            Transformed feature names
        """
        check_is_fitted(self, ["output_features_"])
        return np.array(self.output_features_)
