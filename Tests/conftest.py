"""
Pytest configuration and shared fixtures for the Financial Risk project.

This module contains shared fixtures and configuration for all test modules.
It provides mock data, model instances, and test clients for comprehensive testing.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
from typing import Dict, Any, Tuple
import pickle

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules with fallbacks
try:
    from FeatureEngineering.schema_validator import RAW_FEATURE_SCHEMA
except ImportError:
    # Fallback schema for testing
    RAW_FEATURE_SCHEMA = [
        'ApplicationDate', 'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus',
        'EducationLevel', 'Experience', 'LoanAmount', 'LoanDuration', 'MaritalStatus',
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
    Fixture providing sample loan data matching the expected schema.
    
    Returns:
        pd.DataFrame: Sample loan data with all required columns
    """
    np.random.seed(42)  # For reproducible test data
    n_samples = 100
    
    data = {
        'ApplicationDate': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'Age': np.random.randint(18, 80, n_samples),
        'AnnualIncome': np.random.normal(50000, 15000, n_samples).clip(20000, 200000),
        'CreditScore': np.random.randint(300, 850, n_samples),
        'EmploymentStatus': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples),
        'EducationLevel': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Experience': np.random.randint(0, 40, n_samples),
        'LoanAmount': np.random.normal(25000, 10000, n_samples).clip(1000, 100000),
        'LoanDuration': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'NumberOfDependents': np.random.randint(0, 5, n_samples),
        'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples),
        'MonthlyDebtPayments': np.random.normal(1000, 500, n_samples).clip(0, 5000),
        'CreditCardUtilizationRate': np.random.uniform(0, 1, n_samples),
        'NumberOfOpenCreditLines': np.random.randint(1, 10, n_samples),
        'NumberOfCreditInquiries': np.random.randint(0, 10, n_samples),
        'DebtToIncomeRatio': np.random.uniform(0, 0.8, n_samples),
        'BankruptcyHistory': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'LoanPurpose': np.random.choice(['Home', 'Auto', 'Personal', 'Education'], n_samples),
        'PreviousLoanDefaults': np.random.randint(0, 3, n_samples),
        'PaymentHistory': np.random.uniform(0, 1, n_samples),
        'LengthOfCreditHistory': np.random.randint(1, 30, n_samples),
        'SavingsAccountBalance': np.random.normal(10000, 5000, n_samples).clip(0, 50000),
        'CheckingAccountBalance': np.random.normal(2000, 1000, n_samples).clip(0, 10000),
        'TotalAssets': np.random.normal(50000, 25000, n_samples).clip(0, 200000),
        'TotalLiabilities': np.random.normal(20000, 15000, n_samples).clip(0, 100000),
        'MonthlyIncome': np.random.normal(4000, 1200, n_samples).clip(1500, 15000),
        'UtilityBillsPaymentHistory': np.random.uniform(0, 1, n_samples),
        'JobTenure': np.random.randint(0, 20, n_samples),
        'NetWorth': np.random.normal(30000, 20000, n_samples),
        'BaseInterestRate': np.random.uniform(0.03, 0.08, n_samples),
        'InterestRate': np.random.uniform(0.05, 0.15, n_samples),
        'MonthlyLoanPayment': np.random.normal(500, 200, n_samples).clip(50, 2000),
        'TotalDebtToIncomeRatio': np.random.uniform(0, 1, n_samples)
    }
    
    return pd.DataFrame(data)


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
    Fixture providing a small dataset for quick tests.
    
    Returns:
        pd.DataFrame: Small loan dataset with 10 samples
    """
    np.random.seed(42)
    n_samples = 10
    
    data = {
        'ApplicationDate': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'Age': np.random.randint(25, 65, n_samples),
        'AnnualIncome': np.random.normal(50000, 10000, n_samples).clip(30000, 80000),
        'CreditScore': np.random.randint(600, 800, n_samples),
        'EmploymentStatus': ['Employed'] * n_samples,
        'EducationLevel': ['Bachelor'] * n_samples,
        'Experience': np.random.randint(2, 15, n_samples),
        'LoanAmount': np.random.normal(20000, 5000, n_samples).clip(10000, 40000),
        'LoanDuration': [36] * n_samples,
        'MaritalStatus': ['Married'] * n_samples,
        'NumberOfDependents': np.random.randint(0, 3, n_samples),
        'HomeOwnershipStatus': ['Own'] * n_samples,
        'MonthlyDebtPayments': np.random.normal(800, 200, n_samples).clip(400, 1500),
        'CreditCardUtilizationRate': np.random.uniform(0.1, 0.5, n_samples),
        'NumberOfOpenCreditLines': np.random.randint(2, 6, n_samples),
        'NumberOfCreditInquiries': np.random.randint(0, 3, n_samples),
        'DebtToIncomeRatio': np.random.uniform(0.2, 0.5, n_samples),
        'BankruptcyHistory': [0] * n_samples,
        'LoanPurpose': ['Home'] * n_samples,
        'PreviousLoanDefaults': [0] * n_samples,
        'PaymentHistory': np.random.uniform(0.8, 1.0, n_samples),
        'LengthOfCreditHistory': np.random.randint(5, 20, n_samples),
        'SavingsAccountBalance': np.random.normal(8000, 2000, n_samples).clip(3000, 15000),
        'CheckingAccountBalance': np.random.normal(1500, 500, n_samples).clip(500, 3000),
        'TotalAssets': np.random.normal(40000, 10000, n_samples).clip(20000, 70000),
        'TotalLiabilities': np.random.normal(15000, 5000, n_samples).clip(5000, 30000),
        'MonthlyIncome': np.random.normal(4000, 800, n_samples).clip(2500, 6000),
        'UtilityBillsPaymentHistory': np.random.uniform(0.9, 1.0, n_samples),
        'JobTenure': np.random.randint(2, 10, n_samples),
        'NetWorth': np.random.normal(25000, 8000, n_samples),
        'BaseInterestRate': [0.05] * n_samples,
        'InterestRate': np.random.uniform(0.06, 0.10, n_samples),
        'MonthlyLoanPayment': np.random.normal(400, 100, n_samples).clip(250, 600),
        'TotalDebtToIncomeRatio': np.random.uniform(0.3, 0.6, n_samples)
    }
    
    return pd.DataFrame(data)


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


@pytest.fixture
def flask_test_client():
    """
    Fixture providing a Flask test client for API testing.
    
    Returns:
        Flask test client
    """
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def invalid_loan_data():
    """
    Fixture providing invalid loan data for testing validation.
    
    Returns:
        pd.DataFrame: Invalid loan data missing required columns
    """
    return pd.DataFrame({
        'Age': [25, 30, 35],
        'Income': [50000, 60000, 70000],  # Wrong column name
        'InvalidColumn': [1, 2, 3]  # Extra invalid column
    })


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
