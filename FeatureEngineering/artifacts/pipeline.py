import pandas as pd

class SimplePipeline:
    """A minimal preprocessing pipeline used for testing.

    It ensures that all expected feature columns are present in the DataFrame,
    fills missing columns with a default value of 0, and orders the columns
    according to the expected feature list.
    """
    def __init__(self, expected_features):
        self.expected_features = expected_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add missing columns with default 0
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns
        return df[self.expected_features]
