import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
from .base_transformer import BaseTransformer


class EncodingTransformer(BaseTransformer):
    """
    Applies ordinal and one-hot encoding to categorical features.
    
    Encoding Strategy:
    - EducationLevel (ordinal):
      0 = High School
      1 = Associate
      2 = Bachelor
      3 = Master
      4 = Doctorate
    - Other categoricals (one-hot):
      * HomeOwnershipStatus
      * EmploymentStatus
      * LoanPurpose
    """

    def __init__(self, verbose=False):
        """
        Initialize the transformer with strict ordinal encoding for EducationLevel.
        Will preserve the 0-4 numeric mapping but ensure consistent encoding.
        """
        super().__init__(verbose=verbose)
        self.ordinal_columns = ['EducationLevel']
        self.onehot_columns = ['HomeOwnershipStatus', 'EmploymentStatus', 'LoanPurpose']
        
        # EducationLevel ordinal encoding mapping:
        # 0 = High School
        # 1 = Associate
        # 2 = Bachelor
        # 3 = Master
        # 4 = Doctorate
        # Note: Input must be the original string values
        self.education_categories = [
            'High School',
            'Associate',
            'Bachelor',
            'Master',
            'Doctorate'
        ]
        self.ordinal_encoder = OrdinalEncoder(
            categories=[self.education_categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int8
        )
        self.onehot_encoder = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )
        self.most_frequent_categories = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit encoders and determine output feature names.
        """
        super().fit(X, y)
        self._log_transformation("Fitting encoding transformer...")

        # Identify which columns are present for encoding
        self.ordinal_columns_present_ = [col for col in self.ordinal_columns if col in self.feature_names_in_]
        self.onehot_columns_present_ = [col for col in self.onehot_columns if col in self.feature_names_in_]

        # Fit ordinal encoder
        if self.ordinal_columns_present_:
            self.ordinal_encoder.fit(X[self.ordinal_columns_present_])

        # Fit one-hot encoder and store most frequent categories for imputation
        if self.onehot_columns_present_:
            self.onehot_encoder.fit(X[self.onehot_columns_present_])
            self.onehot_feature_names_ = self.onehot_encoder.get_feature_names_out(self.onehot_columns_present_)
            for col in self.onehot_columns_present_:
                self.most_frequent_categories[col] = X[col].mode().iloc[0]

        # Define final output feature names
        # Start with columns that are not one-hot encoded
        other_cols = [c for c in self.feature_names_in_ if c not in self.onehot_columns_present_ and c not in self.ordinal_columns_present_]
        # Add the new one-hot feature names
        self.feature_names_out_ = other_cols + list(self.onehot_feature_names_ if hasattr(self, 'onehot_feature_names_') else [])

        self._log_transformation("Encoding transformer fitted successfully.")
        return self

    def _transform(self, X):
        """Transform categorical features without imputation."""
        X = X.copy()
        
        # Validate no missing values
        for col in self.ordinal_columns + self.onehot_columns:
            if col in X.columns and X[col].isnull().any():
                raise ValueError(f"Missing values found in {col} - imputation not allowed")
        
        # Ordinal encoding
        if 'EducationLevel' in X.columns:
            X['EducationLevel'] = self.ordinal_encoder.transform(
                X[['EducationLevel']]
            ).astype(int)
            
        # One-hot encoding
        if self.onehot_columns_present_:
            onehot_data = self.onehot_encoder.transform(X[self.onehot_columns_present_])
            onehot_df = pd.DataFrame(
                onehot_data, 
                columns=self.onehot_feature_names_, 
                index=X.index
            )
            # Drop original categorical columns and add one-hot encoded columns
            X = X.drop(columns=self.onehot_columns_present_)
            X = pd.concat([X, onehot_df], axis=1)
            
        # Ensure output columns are in the expected order
        if hasattr(self, 'feature_names_out_'):
            X = X.reindex(columns=self.feature_names_out_)
            
        return X