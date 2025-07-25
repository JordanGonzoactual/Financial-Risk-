import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the exact raw feature schema required for inference
# These are the columns expected in the raw input data from the frontend.
RAW_FEATURE_SCHEMA = [
    'ApplicationDate',
    'Age',
    'AnnualIncome',
    'CreditScore',
    'EmploymentStatus',
    'EducationLevel',
    'Experience',
    'LoanAmount',
    'LoanDuration',
    'MaritalStatus',
    'NumberOfDependents',
    'HomeOwnershipStatus',
    'MonthlyDebtPayments',
    'CreditCardUtilizationRate',
    'NumberOfOpenCreditLines',
    'NumberOfCreditInquiries',
    'DebtToIncomeRatio',
    'BankruptcyHistory',
    'LoanPurpose',
    'PreviousLoanDefaults',
    'PaymentHistory',
    'LengthOfCreditHistory',
    'SavingsAccountBalance',
    'CheckingAccountBalance',
    'TotalAssets',
    'TotalLiabilities',
    'MonthlyIncome',
    'UtilityBillsPaymentHistory',
    'JobTenure',
    'NetWorth',
    'BaseInterestRate',
    'InterestRate',
    'MonthlyLoanPayment',
    'TotalDebtToIncomeRatio'
]


def validate_raw_data_schema(data_df: pd.DataFrame):
    """Validates the schema of the raw input DataFrame against the expected raw features."""
    logging.info("Initiating schema validation for raw inference data...")
    
    try:
        incoming_features = data_df.columns.tolist()
        logging.info(f"Expected columns: {RAW_FEATURE_SCHEMA}")
        logging.info(f"Incoming columns: {incoming_features}")
        
        expected_set = set(RAW_FEATURE_SCHEMA)
        incoming_set = set(incoming_features)
        
        # Check for missing columns
        missing_columns = expected_set - incoming_set
        if missing_columns:
            error_msg = f"Schema validation failed. Missing required columns: {sorted(list(missing_columns))}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        # Check for unexpected columns
        extra_columns = incoming_set - expected_set
        if extra_columns:
            # For raw data, extra columns are usually an error, but we'll just warn.
            logging.warning(f"Schema validation warning. Extra columns found and will be ignored: {sorted(list(extra_columns))}")
            
        logging.info("Raw data schema validation successful. All required columns are present.")
        return True
        
    except ValueError as ve:
        # Re-raise the specific validation error to be caught by the API endpoint
        raise ve
    except Exception as e:
        error_msg = f"An unexpected error occurred during schema validation: {e}"
        logging.error(error_msg)
        # Wrap unexpected errors in a ValueError to be handled consistently
        raise ValueError(error_msg)
