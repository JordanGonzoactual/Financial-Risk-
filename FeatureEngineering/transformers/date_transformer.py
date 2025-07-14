import pandas as pd
from .base_transformer import BaseTransformer


class DateTransformer(BaseTransformer):
    """
    Transformer that extracts year, month, and day components from a date column
    and removes the original date column.
    """

    def __init__(self, date_column='ApplicationDate', verbose=False):
        """
        Initialize the DateTransformer.

        Args:
            date_column: Name of the date column to transform.
            verbose: Whether to log transformation details.
        """
        super().__init__(verbose=verbose)
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        """
        Validate the date column and set up output feature names.
        """
        super().fit(X, y)

        if self.date_column not in self.feature_names_in_:
            raise ValueError(f"Date column '{self.date_column}' not found in input DataFrame")

        try:
            # Validate that the column can be parsed as datetime
            pd.to_datetime(X[self.date_column], errors='raise')
        except Exception as e:
            raise ValueError(f"Failed to parse '{self.date_column}' as datetime: {e}")

        # Define the new features that will be created
        date_col_lower = self.date_column.lower()
        self.new_features_ = [
            f"{date_col_lower}_year",
            f"{date_col_lower}_month",
            f"{date_col_lower}_day"
        ]
        
        # Final output features are input features minus the original date column plus the new ones
        self.feature_names_out_ = [col for col in self.feature_names_in_ if col != self.date_column] + self.new_features_

        self._log_transformation(f"Will extract date components from '{self.date_column}'.")
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date components and remove the original date column.
        """
        X_transformed = X # The base class transform method provides a copy

        # Parse dates, coercing errors to NaT for missing or invalid dates
        dates = pd.to_datetime(X_transformed[self.date_column], errors='coerce')

        # Extract year, month, and day
        date_col_lower = self.date_column.lower()
        X_transformed[f"{date_col_lower}_year"] = dates.dt.year
        X_transformed[f"{date_col_lower}_month"] = dates.dt.month
        X_transformed[f"{date_col_lower}_day"] = dates.dt.day

        # Drop the original date column
        X_transformed = X_transformed.drop(columns=[self.date_column])

        # Convert new date components to a nullable integer type to support NaNs
        for col in self.new_features_:
            X_transformed[col] = X_transformed[col].astype('Int64')

        self._log_transformation(f"Successfully extracted date components from '{self.date_column}'.")
        return X_transformed
