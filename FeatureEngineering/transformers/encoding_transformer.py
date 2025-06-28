import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import logging

class EncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Define ordinal mappings
        self.education_mapping = {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
            'Doctorate': 4
        }
        self.life_stage_mapping = {'Young': 0, 'Middle': 1, 'Senior': 2}
        
        # Define columns for one-hot and ordinal encoding
        self.onehot_columns = ['MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose', 'EmploymentStatus']
        self.ordinal_columns = ['EducationLevel', 'life_stage']
        
        # Initialize encoders
        self.ordinal_encoder = OrdinalEncoder(categories=[
            list(self.education_mapping.keys()),
            list(self.life_stage_mapping.keys())
        ])
        self.onehot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        
        # Keep track of columns present in the training data
        self.ordinal_columns_present = []
        self.onehot_columns_present = []

        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        # Store feature names for `get_feature_names_out`
        self.feature_names_in_ = list(X.columns)
        
        # Check which encoding columns are present in the data
        columns = X.columns
        
        # Check which ordinal columns are present and fit the encoder
        self.ordinal_columns_present = [col for col in self.ordinal_columns if col in columns]
        if self.ordinal_columns_present:
            # We need to filter the categories for the encoder to match the present columns.
            all_categories_map = {
                'EducationLevel': list(self.education_mapping.keys()),
                'life_stage': list(self.life_stage_mapping.keys())
            }
            present_categories = [all_categories_map[col] for col in self.ordinal_columns_present]
            
            # Re-initialize the encoder with the correct categories for the present columns
            temp_ordinal_encoder = OrdinalEncoder(categories=present_categories)
            temp_ordinal_encoder.fit(X[self.ordinal_columns_present])
            self.ordinal_encoder = temp_ordinal_encoder

        # Check which one-hot columns are present
        self.onehot_columns_present = [col for col in self.onehot_columns if col in columns]
        
        if self.onehot_columns_present:
            self.onehot_encoder.fit(X[self.onehot_columns_present])
        
        return self

    def transform(self, X):
        """
        Transforms the data by applying ordinal and one-hot encoding to specified
        columns, and passing through all other columns untouched.
        """
        X_copy = X.copy()  # Work on a copy to avoid side effects

        # Identify columns this transformer will process
        ordinal_to_process = [col for col in self.ordinal_columns_present if col in X_copy.columns]
        onehot_to_process = [col for col in self.onehot_columns_present if col in X_copy.columns]
        all_cols_to_process = ordinal_to_process + onehot_to_process

        # --- Logging for debugging NaN and unseen values ---
        self.logger.info("--- Encoding Transformer: Checking for unexpected values ---")
        if all_cols_to_process:
            nan_in_input = X_copy[all_cols_to_process].isnull().sum()
            if nan_in_input.sum() > 0:
                self.logger.warning(f"NaN values found in input columns to be encoded:\n{nan_in_input[nan_in_input > 0]}")

        if ordinal_to_process:
            all_known_cats = {
                'EducationLevel': set(self.education_mapping.keys()),
                'life_stage': set(self.life_stage_mapping.keys())
            }
            for col in ordinal_to_process:
                if col in all_known_cats:
                    known_cats = all_known_cats[col]
                    unseen_cats = set(X_copy[col].dropna().unique()) - known_cats
                    if unseen_cats:
                        self.logger.warning(f"Unseen categories in '{col}': {unseen_cats}. This will cause an error.")

        if onehot_to_process:
            for col in onehot_to_process:
                try:
                    col_idx = self.onehot_columns_present.index(col)
                    known_cats = set(self.onehot_encoder.categories_[col_idx])
                    unseen_cats = set(X_copy[col].dropna().unique()) - known_cats
                    if unseen_cats:
                        self.logger.warning(f"Unseen categories in '{col}': {unseen_cats}. Using 'handle_unknown=ignore'.")
                except (IndexError, ValueError):
                    self.logger.error(f"Column '{col}' not fitted correctly in OneHotEncoder.")

        # --- Transformation ---
        passthrough_cols = [col for col in X_copy.columns if col not in all_cols_to_process]
        output_dfs = [X_copy[passthrough_cols]]

        if ordinal_to_process:
            ordinal_data = self.ordinal_encoder.transform(X_copy[ordinal_to_process])
            ordinal_df = pd.DataFrame(ordinal_data, columns=ordinal_to_process, index=X_copy.index)
            output_dfs.append(ordinal_df)

        if onehot_to_process:
            onehot_data = self.onehot_encoder.transform(X_copy[onehot_to_process])
            onehot_names = self.onehot_encoder.get_feature_names_out(onehot_to_process)
            onehot_df = pd.DataFrame(onehot_data, columns=onehot_names, index=X_copy.index)
            output_dfs.append(onehot_df)

        X_transformed = pd.concat(output_dfs, axis=1)

        # --- Final NaN check ---
        nan_in_output = X_transformed.isnull().sum().sum()
        if nan_in_output > 0:
            self.logger.error(f"NaNs found in the output of EncodingTransformer! Total: {nan_in_output}")
            nan_cols = X_transformed.isnull().sum()
            self.logger.error(f"Columns with NaNs:\n{nan_cols[nan_cols > 0]}")

        final_column_order = self.get_feature_names_out(list(X.columns))
        return X_transformed.reindex(columns=final_column_order)

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names after transformation, ensuring consistent order.
        Order: passthrough columns, ordinal columns, one-hot encoded columns.
        """
        if input_features is None:
            input_features = self.feature_names_in_

        known_ordinal = self.ordinal_columns_present
        known_onehot = self.onehot_columns_present
        
        ordinal_to_process = [col for col in known_ordinal if col in input_features]
        onehot_to_process = [col for col in known_onehot if col in input_features]
        all_processed = ordinal_to_process + onehot_to_process

        passthrough_cols = [col for col in input_features if col not in all_processed]
        
        ordinal_cols_out = ordinal_to_process

        onehot_cols_out = []
        if onehot_to_process:
            onehot_cols_out = self.onehot_encoder.get_feature_names_out(onehot_to_process).tolist()

        return passthrough_cols + ordinal_cols_out + onehot_cols_out