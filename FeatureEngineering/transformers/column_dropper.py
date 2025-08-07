import pandas as pd
from .base_transformer import BaseTransformer


class ColumnDropper(BaseTransformer):
    """A transformer to drop specified columns from a DataFrame."""

    def __init__(self, columns_to_drop, verbose=False):
        """
        Initialize the transformer.
        Args:
            columns_to_drop (list): A list of column names to drop.
            verbose (bool): If True, enables detailed logging.
        """
        super().__init__(verbose=verbose)
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer, validating columns and setting output feature names.
        """
        super().fit(X, y)
        self._log_transformation(f"Fitting ColumnDropper. Columns to drop: {self.columns_to_drop}")

        # Define output feature names
        self.feature_names_out_ = [col for col in self.feature_names_in_ if col not in self.columns_to_drop]

        self._log_transformation("ColumnDropper fitted successfully.")
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the specified columns from the DataFrame.
        """
        self._log_transformation(f"Dropping columns: {self.columns_to_drop}")

        # Identify columns that actually exist in the dataframe to avoid errors
        cols_to_drop_present = [col for col in self.columns_to_drop if col in X.columns]

        if not cols_to_drop_present:
            self._log_transformation("No specified columns to drop were found in the DataFrame.", level='warning')
            return X

        X_transformed = X.drop(columns=cols_to_drop_present)
        self._log_transformation(f"Successfully dropped columns: {cols_to_drop_present}")

        return X_transformed
