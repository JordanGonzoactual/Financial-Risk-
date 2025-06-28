from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    """A transformer to drop specified columns from a DataFrame."""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """Fit method is a no-op."""
        return self

    def transform(self, X):
        """Drop the specified columns from the DataFrame."""
        return X.drop(columns=self.columns_to_drop, errors='ignore')
