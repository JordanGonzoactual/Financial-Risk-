import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Optional
from .base_transformer import BaseTransformer


class PolynomialTransformer(BaseTransformer):
    """
    Custom transformer that creates polynomial features for specific numerical columns
    while preserving other columns in the dataset.
    """

    def __init__(self, 
                 target_features: Optional[List[str]] = None, 
                 degree: int = 2, 
                 include_bias: bool = False, 
                 interaction_only: bool = False,
                 verbose: bool = False):
        """
        Initialize the PolynomialTransformer.
        
        Args:
            target_features: List of column names to transform. If None, defaults will be used.
            degree: The degree of the polynomial features.
            include_bias: If True, include a bias column (all 1s).
            interaction_only: If True, only include interaction features.
            verbose: If True, enables detailed logging.
        """
        super().__init__(verbose=verbose)
        self.target_features = target_features
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        
        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        self.available_features_ = []
        self.poly_feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer, identify target columns, and set up feature names.
        """
        super().fit(X, y)
        self._log_transformation("Fitting PolynomialTransformer...")

        if self.target_features is None:
            self.target_features = ['AnnualIncome', 'CreditScore', 'TotalAssets', 'TotalLiabilities', 'NetWorth']
            self._log_transformation(f"No target features specified. Defaulting to: {self.target_features}")

        self.available_features_ = [col for col in self.target_features if col in self.feature_names_in_]
        
        missing_features = set(self.target_features) - set(self.available_features_)
        if missing_features:
            self._log_transformation(f"Features not found and will be ignored: {missing_features}", level='warning')
        
        if not self.available_features_:
            self._log_transformation("No target features available in the input data.", level='warning')
            self.feature_names_out_ = self.feature_names_in_
            return self
        
        self.poly.fit(X[self.available_features_])
        self.poly_feature_names_ = self._generate_feature_names()
        
        # Define final output feature names
        non_target_cols = [col for col in self.feature_names_in_ if col not in self.available_features_]
        self.feature_names_out_ = non_target_cols + self.poly_feature_names_

        self._log_transformation("PolynomialTransformer fitted successfully.")
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply polynomial features to target columns and preserve others.
        """
        if not self.available_features_:
            return X

        self._log_transformation(f"Applying polynomial features to {self.available_features_}...")
        
        # Separate target and non-target features
        target_data = X[self.available_features_]
        non_target_cols = [col for col in X.columns if col not in self.available_features_]
        non_target_data = X[non_target_cols].copy() if non_target_cols else pd.DataFrame(index=X.index)
        
        # Transform target features
        poly_features = self.poly.transform(target_data)
        
        poly_df = pd.DataFrame(
            data=poly_features,
            columns=self.poly_feature_names_,
            index=X.index
        )
        
        # Concatenate non-target and polynomial features
        result = pd.concat([non_target_data, poly_df], axis=1)
        self._log_transformation("Polynomial features applied successfully.")
        
        return result

    def _generate_feature_names(self) -> List[str]:
        """
        Generate meaningful feature names for polynomial features.
        """
        raw_feature_names = self.poly.get_feature_names_out(self.available_features_)
        
        readable_names = []
        for name in raw_feature_names:
            if name == '1':
                readable_names.append('bias')
                continue
            
            # Simplified naming for clarity
            name = name.replace(' ', '*')
            readable_names.append(name)
                
        return readable_names
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get output feature names for transformation.
        Parameters:
        -----------
        input_features : ignored
            Not used, present for API consistency
            
        Returns:
        --------
        np.ndarray
            Array of feature names
        """
        if not self.available_features:
            if input_features is not None:
                return np.array(input_features)
            return np.array([])
        
        non_target_cols = [col for col in input_features if col not in self.available_features] if input_features else []
        
        return np.array(non_target_cols + self.poly_feature_names)