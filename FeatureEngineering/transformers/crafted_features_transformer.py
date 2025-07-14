import pandas as pd
import numpy as np
from typing import List, Union, Optional

from .base_transformer import BaseTransformer


class CraftedFeaturesTransformer(BaseTransformer):
    """
    Transformer that creates domain-specific financial features.

    Features created:
    - liquidity_ratio: (SavingsAccountBalance + CheckingAccountBalance) / MonthlyIncome
    - debt_burden: MonthlyDebtPayments / MonthlyIncome
    - payment_capacity: MonthlyIncome - MonthlyDebtPayments - MonthlyLoanPayment
    - loan_to_income_ratio: LoanAmount / AnnualIncome
    """

    def __init__(
        self,
        zero_division_mode: str = "zero",
        verbose: bool = False
    ):
        """
        Initialize the CraftedFeaturesTransformer.

        Args:
            zero_division_mode: How to handle division by zero ('nan', 'zero', or 'inf')
            verbose: Whether to log detailed information during transformation
        """
        super().__init__(verbose=verbose)
        self.zero_division_mode = zero_division_mode

        # Define required input columns
        self.required_columns = [
            'SavingsAccountBalance', 'CheckingAccountBalance',
            'MonthlyIncome', 'MonthlyDebtPayments', 'MonthlyLoanPayment',
            'LoanAmount', 'AnnualIncome'
        ]

        # Define output feature names
        self.feature_names = [
            'liquidity_ratio', 'debt_burden', 'payment_capacity',
            'loan_to_income_ratio'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer and validate required columns.
        """
        super().fit(X, y)
        # Validate that all required columns are present
        missing_cols = set(self.required_columns) - set(self.feature_names_in_)
        if missing_cols:
            raise ValueError(f"Required columns are missing from the input DataFrame: {missing_cols}")

        # The output features will be the input features plus the new ones
        self.feature_names_out_ = self.feature_names_in_ + self.feature_names
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The core transformation logic that creates financial features.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added financial features
        """
        X_transformed = X  # Input is already a copy from the base class
        self._log_transformation("Creating financial features...")

        # Create liquidity_ratio
        X_transformed['liquidity_ratio'] = self._safe_divide(
            (X_transformed['SavingsAccountBalance'] + X_transformed['CheckingAccountBalance']),
            X_transformed['MonthlyIncome'],
            'liquidity_ratio'
        )

        # Create debt_burden
        X_transformed['debt_burden'] = self._safe_divide(
            X_transformed['MonthlyDebtPayments'],
            X_transformed['MonthlyIncome'],
            'debt_burden'
        )

        # Create payment_capacity
        X_transformed['payment_capacity'] = (
            X_transformed['MonthlyIncome'] -
            X_transformed['MonthlyDebtPayments'] -
            X_transformed['MonthlyLoanPayment']
        )

        # Create loan_to_income_ratio
        X_transformed['loan_to_income_ratio'] = self._safe_divide(
            X_transformed['LoanAmount'],
            X_transformed['AnnualIncome'],
            'loan_to_income_ratio'
        )



        # Optimize dtypes of new features
        X_transformed = self._optimize_dtypes(X_transformed)

        self._log_transformation("Successfully created financial features.")

        return X_transformed

    def _safe_divide(self, numerator: Union[pd.Series, float],
                     denominator: Union[pd.Series, float],
                     feature_name: str) -> pd.Series:
        """
        Safely handle division by zero based on specified mode, with logging.

        Args:
            numerator: The dividend in the division.
            denominator: The divisor in the division.
            feature_name: The name of the feature being calculated, for logging.

        Returns:
            A pandas Series with the division result.
        """
        # Find where the denominator is zero
        denominator_series = denominator if isinstance(denominator, pd.Series) else pd.Series([denominator])
        zero_denominator_mask = (denominator_series == 0)

        # Log a warning if any zeros are found
        if zero_denominator_mask.any():
            zero_count = zero_denominator_mask.sum()
            self._log_transformation(
                f"Found {zero_count} instances of division by zero when calculating '{feature_name}'. "
                f"Handling with mode: '{self.zero_division_mode}'.",
                level="warning"
            )

        # Perform division based on the selected mode
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.zero_division_mode == "zero":
                result = np.where(zero_denominator_mask, 0, numerator / denominator)
            elif self.zero_division_mode == "inf":
                result = np.where(zero_denominator_mask, np.inf, numerator / denominator)
            else:  # default to "nan"
                result = np.where(zero_denominator_mask, np.nan, numerator / denominator)

        index = numerator.index if isinstance(numerator, pd.Series) else denominator.index
        return pd.Series(result, index=index)

    def _optimize_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for the crafted features.

        Args:
            X: DataFrame with crafted features

        Returns:
            DataFrame with optimized data types
        """
        # Convert numeric features to appropriate types
        numeric_features = ['liquidity_ratio', 'debt_burden',
                           'payment_capacity', 'loan_to_income_ratio']

        for feature in numeric_features:
            if feature in X.columns:
                # Use float32 for ratio features
                if feature.endswith('_ratio') or feature == 'debt_burden':
                    X[feature] = X[feature].astype('float32')
                # Use int32 for payment_capacity if possible
                elif feature == 'payment_capacity' and not X[feature].isna().any():
                    X[feature] = X[feature].astype('int32')

        # Ensure life_stage is properly categorical with ordered categories
        if 'life_stage' in X.columns:
            X['life_stage'] = pd.Categorical(
                X['life_stage'],
                categories=['Young', 'Middle', 'Senior'],
                ordered=True
            )

        return X 