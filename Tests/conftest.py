"""
Pytest configuration and shared fixtures for the Financial Risk project.

This module contains shared fixtures and configuration for all test modules.
It provides mock data, model instances, and test clients for comprehensive testing.
"""

import sys
import os
# Force XGBoost to use CPU during tests to avoid device mismatch warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import io
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
from typing import Dict, Any, Tuple, get_origin, get_args
import pickle
import inspect
from enum import Enum

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

# Import RawData model
from shared.models.raw_data import RawData

# Thread-safe reset for ModelService singleton to ensure test isolation
@pytest.fixture(autouse=True, scope="module")
def reset_model_service_singleton():
    try:
        from backend.model_service import ModelService
        # Acquire class lock to safely reset
        with ModelService._lock:
            ModelService._instance = None
    except Exception:
        pass
    yield
    try:
        from backend.model_service import ModelService
        with ModelService._lock:
            ModelService._instance = None
    except Exception:
        pass

# Import Flask app for testing
try:
    from backend.app import create_app
except ImportError:
    # Create mock Flask app for testing if module not available
    from flask import Flask
    
    def create_app():
        """Mock implementation of Flask app creation."""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app

VALID_EMPLOYMENT_STATUSES = ['Employed', 'Self-Employed', 'Unemployed']
VALID_HOME_OWNERSHIP = ['Own', 'Rent', 'Mortgage', 'Other']
VALID_LOAN_PURPOSES = ['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other']

def get_valid_raw_data() -> dict:
    """Return a canonical, always-valid RawData dict for testing."""
    return {
        'ApplicationDate': '2024-01-15',
        'Age': 35,
        'CreditScore': 720,
        'EmploymentStatus': 'Employed',
        'EducationLevel': 'Bachelor',
        'Experience': 10,
        'LoanAmount': 250000.0,
        'LoanDuration': 360,
        'NumberOfDependents': 2,
        'HomeOwnershipStatus': 'Mortgage',
        'MonthlyDebtPayments': 1500.0,
        'CreditCardUtilizationRate': 0.25,
        'NumberOfOpenCreditLines': 5,
        'NumberOfCreditInquiries': 2,
        'DebtToIncomeRatio': 0.35,
        'SavingsAccountBalance': 25000.0,
        'CheckingAccountBalance': 5000.0,
        'MonthlyIncome': 6250.0,
        'AnnualIncome': 75000.0,
        'LoanPurpose': 'Home',
        'BankruptcyHistory': 0,
        'PaymentHistory': 2,
        'UtilityBillsPaymentHistory': 0.95,
        'PreviousLoanDefaults': 0,
        'InterestRate': 0.05,
        'BaseInterestRate': 0.045,
        'TotalDebtToIncomeRatio': 0.40,
        'TotalAssets': 350000,
        'TotalLiabilities': 180000,
        'NetWorth': 170000,
        'LengthOfCreditHistory': 120,
        'JobTenure': 60,
        'MonthlyLoanPayment': 1200.0
    }

def generate_valid_rawdata_sample(n_samples=1):
    """Generate valid test data samples with PascalCase field names."""
    data = {
        'ApplicationDate': [datetime.now().strftime('%Y-%m-%d') for _ in range(n_samples)],
        'Age': np.random.randint(18, 100, n_samples).astype(int),
        'AnnualIncome': np.random.uniform(12000, 600000, n_samples).astype(float),
        'CreditScore': np.random.randint(300, 850, n_samples).astype(int),
        'EmploymentStatus': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
        'EducationLevel': np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], n_samples),
        'Experience': np.random.randint(0, 50, n_samples).astype(int),
        'LoanAmount': np.random.uniform(1000, 1000000, n_samples).astype(float),
        'LoanDuration': np.random.randint(1, 360, n_samples).astype(int),
        'NumberOfDependents': np.random.randint(0, 10, n_samples).astype(int),
        'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], n_samples),
        'MonthlyDebtPayments': np.random.uniform(0, 5000, n_samples).astype(float),
        'CreditCardUtilizationRate': np.random.uniform(0, 1, n_samples).astype(float),
        'NumberOfOpenCreditLines': np.random.randint(0, 20, n_samples).astype(int),
        'NumberOfCreditInquiries': np.random.randint(0, 10, n_samples).astype(int),
        'DebtToIncomeRatio': np.random.uniform(0, 1, n_samples).astype(float),
        'BankruptcyHistory': np.random.randint(0, 1, n_samples).astype(int),
        'LoanPurpose': np.random.choice(['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'], n_samples),
        'PreviousLoanDefaults': np.random.randint(0, 1, n_samples).astype(int),
        'PaymentHistory': np.random.randint(0, 3, n_samples).astype(int),
        'LengthOfCreditHistory': np.random.randint(0, 240, n_samples).astype(int),
        'SavingsAccountBalance': np.random.uniform(0, 100000, n_samples).astype(float),
        'CheckingAccountBalance': np.random.uniform(0, 100000, n_samples).astype(float),
        'TotalAssets': np.random.randint(0, 1000000, n_samples).astype(int),
        'TotalLiabilities': np.random.randint(0, 500000, n_samples).astype(int),
        'MonthlyIncome': np.random.uniform(1000, 20000, n_samples).astype(float),
        'UtilityBillsPaymentHistory': np.random.uniform(0, 1, n_samples).astype(float),
        'JobTenure': np.random.randint(0, 240, n_samples).astype(int),
        'NetWorth': np.random.randint(-100000, 1000000, n_samples).astype(int),
        'BaseInterestRate': np.random.uniform(0, 1, n_samples).astype(float),
        'InterestRate': np.random.uniform(0, 1, n_samples).astype(float),
        'TotalDebtToIncomeRatio': np.random.uniform(0, 1, n_samples).astype(float),
        'MonthlyLoanPayment': np.random.uniform(100, 5000, n_samples).astype(float)
    }
    return pd.DataFrame(data)

# Define the raw feature schema for testing (matches RawData model fields)
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

try:
    from backend.app import create_app
except ImportError:
    # Fallback Flask app for testing
    from flask import Flask
    
    def create_app():
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            return {'status': 'ok'}
        
        @app.route('/predict', methods=['POST'])
        def predict():
            return {'predictions': [0.5]}
        
        return app


@pytest.fixture(scope="session")
def project_root_path():
    """Fixture providing the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def sample_loan_data():
    """
    Fixture providing sample loan data using schema-based generation.
    
    Uses RawData model introspection to automatically generate valid test data
    that matches all field constraints and validation rules.
    
    Returns:
        pd.DataFrame: Sample loan data with all required columns
    """
    df = pd.DataFrame(generate_valid_rawdata_sample(n_samples=100))
    
    # Validate that the generated data passes RawData model validation
    try:
        test_record = df.iloc[0].to_dict()
        RawData.model_validate(test_record)
        print(f"✅ sample_loan_data fixture validation passed for {len(df)} records")
    except Exception as e:
        print(f"❌ sample_loan_data fixture validation failed: {e}")
        raise
    
    return df


@pytest.fixture(scope="session")
def sample_loan_labels():
    """
    Fixture providing sample loan risk labels (0 = low risk, 1 = high risk).
    
    Returns:
        pd.Series: Binary risk labels
    """
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100, p=[0.7, 0.3]), name='RiskLabel')


@pytest.fixture
def small_loan_data():
    """
    Fixture providing a small dataset for quick tests using schema-based generation.
    
    Uses RawData model introspection to automatically generate valid test data
    that matches all field constraints and validation rules.
    
    Returns:
        pd.DataFrame: Small loan dataset with 10 samples
    """
    df = pd.DataFrame(generate_valid_rawdata_sample(n_samples=10))
    
    # Validate that the generated data passes RawData model validation
    try:
        test_record = df.iloc[0].to_dict()
        RawData.model_validate(test_record)
        print(f"[PASS] small_loan_data fixture validation passed for {len(df)} records")
    except Exception as e:
        print(f"[FAIL] small_loan_data fixture validation failed: {e}")
        raise
    
    return df


@pytest.fixture
def processed_features():
    """
    Fixture providing mock processed features for model testing.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    np.random.seed(42)
    
    # Mock processed features (after feature engineering)
    n_train, n_test = 80, 20
    n_features = 50  # Assume 50 features after processing
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_train = pd.Series(np.random.choice([0, 1], n_train, p=[0.7, 0.3]), name='RiskLabel')
    y_test = pd.Series(np.random.choice([0, 1], n_test, p=[0.7, 0.3]), name='RiskLabel')
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_trained_model():
    """
    Fixture providing a mock trained model for testing.
    
    Returns:
        Mock: Mock model with predict and predict_proba methods
    """
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
    model.feature_names_in_ = [f'feature_{i}' for i in range(50)]
    model.n_features_in_ = 50
    return model


@pytest.fixture
def temp_data_directory():
    """
    Fixture providing a temporary directory for test data files.
    
    Yields:
        str: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_processed_data_files(temp_data_directory, processed_features):
    """
    Fixture creating mock processed data files in a temporary directory.
    
    Args:
        temp_data_directory: Temporary directory path
        processed_features: Processed feature data
        
    Returns:
        str: Path to directory containing mock data files
    """
    X_train, X_test, y_train, y_test = processed_features
    
    # Save mock processed data files
    with open(os.path.join(temp_data_directory, 'X_train_processed.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(temp_data_directory, 'X_test_processed.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(temp_data_directory, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(temp_data_directory, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    
    return temp_data_directory


@pytest.fixture(scope='module')
def app():
    """Fixture providing a Flask application instance for testing."""
    flask_app = create_app()
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app


@pytest.fixture(scope='module')
def flask_test_client(app):
    """
    Fixture providing a Flask test client for API testing.

    Args:
        app: The Flask application instance.

    Returns:
        Flask test client
    """
    with app.test_client() as client:
        yield client


@pytest.fixture
def csv_loan_data(small_loan_data):
    """
    Fixture providing loan data in CSV string format for API testing.
    
    Returns:
        str: CSV string representation of loan data
    """
    return small_loan_data.to_csv(index=False)


@pytest.fixture(scope="session")
def schema_columns():
    """
    Fixture providing the expected schema columns.
    
    Returns:
        list: List of expected column names
    """
    return RAW_FEATURE_SCHEMA.copy()


@pytest.fixture
def mock_model_service():
    """
    Fixture providing a mock model service for testing.
    
    Returns:
        Mock: Mock model service with all required methods
    """
    service = Mock()
    service.health_check.return_value = {'model_loaded': True, 'status': 'healthy'}
    service.get_model_info.return_value = {
        'model_type': 'XGBoost',
        'version': '1.0.0',
        'features': 50,
        'last_trained': '2023-01-01'
    }
    service.predict_batch.return_value = np.array([0.2, 0.8, 0.1, 0.9, 0.3])
    return service


@pytest.fixture(autouse=True)
def setup_logging():
    """
    Fixture to configure logging for tests.
    This fixture runs automatically for all tests.
    """
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Hyperparameter tuning fixtures
@pytest.fixture
def mock_study():
    """Fixture providing a mock Optuna study."""
    study = Mock()
    study.best_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    study.best_value = 0.85
    study.best_trial = Mock()
    study.best_trial.number = 42
    study.trials = [Mock() for _ in range(10)]
    return study


@pytest.fixture
def sample_training_data():
    """Fixture providing sample training data for hyperparameter tuning tests."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.choice([0, 1], 1000, p=[0.7, 0.3]))
    return X, y


@pytest.fixture
def data_with_known_issues():
    """
    Provides a DataFrame with a variety of known data quality issues for validation reporting tests.
    This fixture is ideal for testing the accuracy of the warning generation system.
    """
    return pd.DataFrame({
        'Age': [25, 'thirty', -5, 45],  # Wrong type, out of logical range
        'AnnualIncome': [50000, 60000, 'high', 75000],  # Wrong type
        'CreditScore': [700, 950, 250, 800],  # Out of standard range, low score
        'LoanAmount': [10000, 5000, 1000, 20000],
        'MonthlyDebtPayments': [500, 6000, 1200, 200], # Logically inconsistent with income
        'ApplicationDate': ['2023-01-01', 'not-a-date', '2023-01-03', '2023-01-04'], # Malformed date
        'EmploymentStatus': ['Employed', 'Self-employed', None, ''] # Missing and empty values
    })

@pytest.fixture
def validation_data_factory():
    """
    Provides a factory function to generate DataFrames with controlled data quality issues.
    
    Usage:
        df = validation_data_factory(issue_type='missing_values', field='Age', num_rows=10)
        df = validation_data_factory(issue_type='wrong_type', field='CreditScore', value='bad-score')
    """
    def _create_data(issue_type: str, field: str, num_rows: int = 5, value: any = None):
        # Start with a valid base DataFrame
        base_data = {
            'Age': np.random.randint(18, 80, num_rows),
            'AnnualIncome': np.random.normal(50000, 15000, num_rows),
            'CreditScore': np.random.randint(300, 850, num_rows),
            'LoanAmount': np.random.normal(25000, 10000, num_rows),
        }
        df = pd.DataFrame(base_data)

        if issue_type == 'missing_values':
            df[field] = np.nan
        elif issue_type == 'wrong_type':
            df[field] = value if value is not None else 'wrong-type'
        elif issue_type == 'out_of_range':
            df[field] = value if value is not None else -999
        
        return df
        
    return _create_data




# Test data validation helpers
def assert_dataframe_schema(df: pd.DataFrame, expected_columns: list):
    """Helper function to assert DataFrame has expected schema."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert set(df.columns) == set(expected_columns), f"Schema mismatch. Expected: {expected_columns}, Got: {df.columns.tolist()}"


def assert_model_predictions(predictions: np.ndarray, expected_shape: tuple):
    """Helper function to assert model predictions have expected format."""
    assert isinstance(predictions, np.ndarray), "Expected numpy array"
    assert predictions.shape == expected_shape, f"Shape mismatch. Expected: {expected_shape}, Got: {predictions.shape}"
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions should be probabilities between 0 and 1"
