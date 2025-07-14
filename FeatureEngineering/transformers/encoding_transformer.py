import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from .base_transformer import BaseTransformer


class EncodingTransformer(BaseTransformer):
    """
    Applies ordinal and one-hot encoding to categorical features.
    - Ordinal encoding is applied to 'EducationLevel'.
    - One-hot encoding is applied to other specified categorical columns.
    """

    def __init__(self, verbose=False):
        """
        Initialize the transformer.
        Args:
            verbose: If True, enables detailed logging.
        """
        super().__init__(verbose=verbose)
        self.education_mapping = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
        self.ordinal_columns = ['EducationLevel']
        self.onehot_columns = ['MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose', 'EmploymentStatus']

        # Configure encoders to handle potential issues in test data
        self.ordinal_encoder = OrdinalEncoder(
            categories=[self.education_mapping],
            handle_unknown='use_encoded_value',
            unknown_value=-1  # Use -1 for unseen categories
        )
        self.onehot_encoder = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',  # Ignore unseen categories in test data
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
        other_cols = [c for c in self.feature_names_in_ if c not in self.onehot_columns_present_]
        # Add the new one-hot feature names
        self.feature_names_out_ = other_cols + list(self.onehot_feature_names_ if hasattr(self, 'onehot_feature_names_') else [])

        self._log_transformation("Encoding transformer fitted successfully.")
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ordinal and one-hot encoding.
        """
        X_transformed = X

        # Handle ordinal columns
        if self.ordinal_columns_present_:
            self._log_transformation(f"Applying ordinal encoding to {self.ordinal_columns_present_}...")
            X_transformed[self.ordinal_columns_present_] = self.ordinal_encoder.transform(X[self.ordinal_columns_present_])

        # Handle one-hot columns
        if self.onehot_columns_present_:
            self._log_transformation(f"Applying one-hot encoding to {self.onehot_columns_present_}...")
            # Impute NaNs with the most frequent category before transforming
            for col in self.onehot_columns_present_:
                if X_transformed[col].isnull().any():
                    fill_value = self.most_frequent_categories[col]
                    self._log_transformation(f"Imputing NaNs in '{col}' with most frequent value: '{fill_value}'.", level='warning')
                    X_transformed[col] = X_transformed[col].fillna(fill_value)

            # Apply one-hot encoding
            onehot_data = self.onehot_encoder.transform(X_transformed[self.onehot_columns_present_])
            onehot_df = pd.DataFrame(onehot_data, columns=self.onehot_feature_names_, index=X_transformed.index)

            # Drop original one-hot columns and concatenate new ones
            X_transformed = X_transformed.drop(columns=self.onehot_columns_present_)
            X_transformed = pd.concat([X_transformed, onehot_df], axis=1)

        self._log_transformation("Categorical features encoded successfully.")
        return X_transformed