import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging
from typing import List, Optional, Union

class PolynomialTransformer:
    """
    Custom transformer that creates polynomial features for specific numerical columns
    while preserving other columns in the dataset.
    """
    
    def __init__(self, 
                 target_features: Optional[List[str]] = None, 
                 degree: int = 2, 
                 include_bias: bool = False, 
                 interaction_only: bool = False):
        """
        Initialize the PolynomialTransformer.
        
        Parameters:
        -----------
        target_features : Optional[List[str]], default=None
            List of column names to transform. If None, all numerical columns will be used.
        degree : int, default=2
            The degree of the polynomial features.
        include_bias : bool, default=False
            If True, include a bias column (all 1s).
        interaction_only : bool, default=False
            If True, only include interaction features.
        """
        self.target_features = target_features
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        
        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        self.available_features = []
        self.poly_feature_names = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer. If target_features is None, it identifies all numerical
        columns. It then fits the polynomial transformer on the available target features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input data.
        y : ignored
            Not used, for API consistency.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        if self.target_features is None:
            self.target_features = ['AnnualIncome', 'CreditScore', 'LoanAmount', 'TotalAssets', 'TotalLiabilities', 'NetWorth']
            logging.info(f"No target features specified. Defaulting to: {self.target_features}")

        self.available_features = [col for col in self.target_features if col in X.columns]
        
        missing_features = set(self.target_features) - set(self.available_features)
        for feature in missing_features:
            logging.warning(f"Feature '{feature}' not found in the input data and will be ignored.")
        
        if not self.available_features:
            logging.warning("None of the specified target features are available in the input data.")
            return self
        
        self.poly.fit(X[self.available_features])
        self.poly_feature_names = self._generate_feature_names()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by applying polynomial features to target columns
        and preserving other columns.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input data
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with polynomial features
        """
        if not self.available_features:
            return X.copy()
        
        # Separate target and non-target features
        target_data = X[self.available_features]
        non_target_cols = [col for col in X.columns if col not in self.available_features]
        non_target_data = X[non_target_cols].copy() if non_target_cols else pd.DataFrame(index=X.index)
        
        # Transform target features
        poly_features = self.poly.transform(target_data)
        
        # Convert to DataFrame with meaningful column names
        poly_df = pd.DataFrame(
            data=poly_features,
            columns=self.poly_feature_names,
            index=X.index
        )
        
        # Optimize data types
        for col in poly_df.columns:
            if poly_df[col].dtype == np.float64:
                poly_df[col] = pd.to_numeric(poly_df[col], downcast='float')
        
        # Concatenate non-target and polynomial features
        result = pd.concat([non_target_data, poly_df], axis=1)
        
        return result
    
    def _generate_feature_names(self) -> List[str]:
        """
        Generate meaningful feature names for polynomial features.
        
        Returns:
        --------
        List[str]
            List of feature names for the polynomial features
        """
        # Get the raw feature names from sklearn
        raw_feature_names = self.poly.get_feature_names_out(self.available_features)
        
        # Create more readable feature names
        readable_names = []
        for name in raw_feature_names:
            if name == '1':
                readable_names.append('bias')
                continue
                
            terms = name.split()
            if len(terms) == 1:
                # Single feature with power 1
                readable_names.append(name)
            else:
                feature_counts = {}
                for term in terms:
                    feature = term.split('^')[0]
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
                parts = []
                for feature, count in feature_counts.items():
                    if count == 1:
                        parts.append(feature)
                    else:
                        parts.append(f"{feature}^{count}")
                
                readable_names.append('_'.join(parts))
        
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