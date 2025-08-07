import pandas as pd
import sys
import os

# Add project root to the Python path to enable module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Define required columns for CSV validation (matches RawData model)
RAW_FEATURE_SCHEMA = [
    'ApplicationDate', 'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus',
    'EducationLevel', 'Experience', 'LoanAmount', 'LoanDuration',
    'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments',
    'CreditCardUtilizationRate', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
    'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose', 'PreviousLoanDefaults',
    'PaymentHistory', 'LengthOfCreditHistory', 'SavingsAccountBalance',
    'CheckingAccountBalance', 'TotalAssets', 'TotalLiabilities', 'MonthlyIncome',
    'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth', 'BaseInterestRate',
    'InterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio'
]

class CSVValidator:
    REQUIRED_COLUMNS = RAW_FEATURE_SCHEMA

    def __init__(self, df):
        self.df = df.copy()
        self.errors = []
        self.warnings = []

    def validate(self):
        self._validate_structure()
        if not self.errors: # Only proceed if structure is valid
            self._clean_data()
            self._validate_data_types()
            self._validate_business_rules()
        return self.generate_validation_report()

    def _validate_structure(self):
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            self.errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    def _clean_data(self):
        # Define column types based on the new schema
        numeric_cols = [
            'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount', 'LoanDuration',
            'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
            'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
            'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
            'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'JobTenure', 'NetWorth',
            'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio'
        ]
        categorical_cols = [
            'EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus',
            'BankruptcyHistory', 'LoanPurpose', 'PreviousLoanDefaults', 'PaymentHistory',
            'UtilityBillsPaymentHistory'
        ]
        date_cols = ['ApplicationDate']

        # Clean numeric columns
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Clean categorical columns
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip()

        # Clean date columns
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def _validate_data_types(self):
        # Check for NaNs created by coercion
        if self.df.isnull().sum().sum() > 0:
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    self.warnings.append(f"Column '{col}' contains non-numeric or missing values.")

    def _validate_business_rules(self):
        # Age validation
        self._check_range('Age', 18, 80, "is outside the valid range (18-80)")
        # CreditScore validation
        self._check_range('CreditScore', 300, 850, "is outside the valid range (300-850)")
        # Positive value checks
        for col in ['AnnualIncome', 'LoanAmount', 'MonthlyIncome']:
            self._check_positive(col, "must be a positive value")
        # Non-negative value checks
        for col in ['Experience', 'LoanDuration', 'NumberOfDependents', 'MonthlyDebtPayments',
                    'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'LengthOfCreditHistory',
                    'JobTenure']:
            self._check_non_negative(col, "cannot be negative")
        # Rate/Ratio checks (0-100% range for simplicity)
        for col in ['CreditCardUtilizationRate', 'InterestRate', 'BaseInterestRate']:
            self._check_range(col, 0, 100, "should be between 0 and 100")

    def _check_range(self, col, min_val, max_val, message):
        if col in self.df.columns:
            errors = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            for i in errors.index:
                self.errors.append(f"Row {i+2}: {col} ({self.df.at[i, col]}) {message}.")

    def _check_positive(self, col, message):
        if col in self.df.columns:
            errors = self.df[self.df[col] <= 0]
            for i in errors.index:
                self.errors.append(f"Row {i+2}: {col} ({self.df.at[i, col]}) {message}.")

    def _check_non_negative(self, col, message):
        if col in self.df.columns:
            errors = self.df[self.df[col] < 0]
            for i in errors.index:
                self.errors.append(f"Row {i+2}: {col} ({self.df.at[i, col]}) {message}.")

    def generate_validation_report(self):
        report = {
            'is_valid': not self.errors,
            'errors': self.errors,
            'warnings': self.warnings,
            'missing_values': self.df.isnull().sum().to_dict()
        }
        return report

    def get_clean_data(self):
        return self.handle_missing_values()

    def handle_missing_values(self):
        # Example: Fill with median for numeric and mode for categorical
        clean_df = self.df.copy()
        for col in clean_df.columns:
            if clean_df[col].dtype == 'number':
                clean_df[col].fillna(clean_df[col].median(), inplace=True)
            else:
                clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
        return clean_df
