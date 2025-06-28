import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging

class CraftedFeaturesTransformer:
    """
    Transformer that creates domain-specific financial features.
    
    Features created:
    - liquidity_ratio: (SavingsAccountBalance + CheckingAccountBalance) / MonthlyIncome
    - debt_burden: MonthlyDebtPayments / MonthlyIncome
    - payment_capacity: MonthlyIncome - MonthlyDebtPayments - MonthlyLoanPayment
    - loan_to_income_ratio: LoanAmount / AnnualIncome
    - life_stage: Categorical feature based on age ranges
    """
    
    def __init__(
        self,
        age_bins: Optional[List[int]] = None,
        zero_division_mode: str = "nan",
        verbose: bool = False
    ):
        """
        Initialize the CraftedFeaturesTransformer.
        
        Args:
            age_bins: Custom age bins for life stage categorization. Default: [18, 35, 55, 120]
            zero_division_mode: How to handle division by zero ('nan', 'zero', or 'inf')
            verbose: Whether to log detailed information during transformation
        """
        self.age_bins = age_bins if age_bins is not None else [18, 35, 55, 120]
        self.zero_division_mode = zero_division_mode
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Define required input columns
        self.required_columns = [
            'Age', 'SavingsAccountBalance', 'CheckingAccountBalance',
            'MonthlyIncome', 'MonthlyDebtPayments', 'MonthlyLoanPayment',
            'LoanAmount', 'AnnualIncome'
        ]
        
        # Define output feature names
        self.feature_names = [
            'liquidity_ratio', 'debt_burden', 'payment_capacity',
            'loan_to_income_ratio', 'life_stage'
        ]
    
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
        zero_denominator_mask = (denominator == 0)
        
        # Log a warning if any zeros are found
        if zero_denominator_mask.any():
            zero_count = zero_denominator_mask.sum()
            self.logger.warning(
                f"Found {zero_count} instances of division by zero when calculating '{feature_name}'. "
                f"Handling with mode: '{self.zero_division_mode}'."
            )

        # Perform division based on the selected mode
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.zero_division_mode == "zero":
                result = np.where(zero_denominator_mask, 0, numerator / denominator)
            elif self.zero_division_mode == "inf":
                result = np.where(zero_denominator_mask, np.inf, numerator / denominator)
            else:  # default to "nan"
                result = np.where(zero_denominator_mask, np.nan, numerator / denominator)
                
        return pd.Series(result, index=numerator.index)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate that all required columns are present in the input data.
        
        Args:
            X: Input DataFrame
            
        Raises:
            ValueError: If any required columns are missing
        """
        missing_cols = [col for col in self.required_columns if col not in X.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating financial features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with added financial features
        """
        try:
            # Make a copy to avoid modifying the original
            X_transformed = X.copy()
            
            # Validate input columns
            self._validate_input(X_transformed)
            
            if self.verbose:
                self.logger.info("Creating financial features...")
            
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
            
            # Create life_stage feature
            age_labels = ['Young', 'Middle', 'Senior']
            X_transformed['life_stage'] = pd.cut(
                X_transformed['Age'],
                bins=self.age_bins,
                labels=age_labels,
                right=False
            )
            
            # Convert to categorical with proper ordering
            X_transformed['life_stage'] = pd.Categorical(
                X_transformed['life_stage'],
                categories=age_labels,
                ordered=True
            )
            
            # Optimize data types
            X_transformed = self._optimize_dtypes(X_transformed)
            
            if self.verbose:
                self.logger.info("Feature creation complete.")
                
            return X_transformed
            
        except Exception as e:
            self.logger.error(f"Error in feature transformation: {str(e)}")
            raise
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method (no-op, included for compatibility with scikit-learn).
        
        Args:
            X: Input DataFrame
            y: Target variable (not used)
            
        Returns:
            self
        """
        return self
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit and transform the data.
        
        Args:
            X: Input DataFrame
            y: Target variable (not used)
            
        Returns:
            DataFrame with added financial features
        """
        return self.transform(X)
    

    def get_feature_names(self) -> List[str]:
        """
        Get the names of features created by this transformer.
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required input columns.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def set_params(self, **params):
        """
        Set parameters for the transformer.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return self
    
    def get_params(self, deep=True) -> Dict:
        """
        Get parameters for the transformer.
        
        Args:
            deep: Whether to return nested parameters
            
        Returns:
            Dictionary of parameters
        """
        return {
            'age_bins': self.age_bins,
            'zero_division_mode': self.zero_division_mode,
            'verbose': self.verbose
        }
    
    def describe_features(self) -> Dict[str, str]:
        """
        Get descriptions of the features created by this transformer.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'liquidity_ratio': 'Sum of savings and checking account balances divided by monthly income',
            'debt_burden': 'Monthly debt payments divided by monthly income',
            'payment_capacity': 'Remaining monthly income after debt and loan payments',
            'loan_to_income_ratio': 'Loan amount divided by annual income',
            'life_stage': 'Categorical feature based on age ranges (Young, Middle, Senior)'
        } 