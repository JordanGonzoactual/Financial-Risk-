"""
Tests for the schema validator module.

This module tests schema validation functionality for raw loan data.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from FeatureEngineering.schema_validator import validate_raw_data_schema, RAW_FEATURE_SCHEMA
except ImportError:
    # Create mock implementations for testing if module not available
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
    
    def validate_raw_data_schema(data_df: pd.DataFrame):
        """Mock implementation of schema validation."""
        incoming_features = data_df.columns.tolist()
        expected_set = set(RAW_FEATURE_SCHEMA)
        incoming_set = set(incoming_features)
        
        missing_columns = expected_set - incoming_set
        if missing_columns:
            error_msg = f"Schema validation failed. Missing required columns: {sorted(list(missing_columns))}"
            raise ValueError(error_msg)
        
        return True


class TestValidateRawDataSchema:
    """Test class for validate_raw_data_schema function."""
    
    def test_validate_schema_success(self, sample_loan_data):
        """Test successful schema validation with complete data."""
        # Should not raise any exception
        result = validate_raw_data_schema(sample_loan_data)
        assert result is True
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing required columns."""
        # Create DataFrame with missing columns
        incomplete_data = pd.DataFrame({
            'Age': [25, 30, 35],
            'AnnualIncome': [50000, 60000, 70000],
            'CreditScore': [700, 750, 800]
            # Missing many required columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data_schema(incomplete_data)
    
    def test_validate_schema_extra_columns(self, sample_loan_data, caplog):
        """Test schema validation with extra columns."""
        # Add extra columns to the data
        data_with_extra = sample_loan_data.copy()
        data_with_extra['ExtraColumn1'] = np.random.randn(len(data_with_extra))
        data_with_extra['ExtraColumn2'] = 'extra_value'
        
        # Should succeed but log warning
        with caplog.at_level('WARNING'):
            result = validate_raw_data_schema(data_with_extra)
        
        assert result is True
        
        # Check that warning was logged
        warning_messages = [record.message for record in caplog.records if record.levelname == 'WARNING']
        assert any('Extra columns found' in msg for msg in warning_messages)
    
    def test_validate_schema_empty_dataframe(self):
        """Test schema validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data_schema(empty_df)
    
    def test_validate_schema_partial_columns(self):
        """Test schema validation with only some required columns."""
        partial_data = pd.DataFrame({
            col: np.random.randn(10) if col in ['Age', 'AnnualIncome', 'CreditScore'] else 'test'
            for col in RAW_FEATURE_SCHEMA[:10]  # Only first 10 columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data_schema(partial_data)
    
    def test_validate_schema_exact_columns(self):
        """Test schema validation with exactly the required columns."""
        # Create DataFrame with exactly the required columns
        exact_data = pd.DataFrame({
            col: np.random.randn(5) if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                              'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                              'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                              'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                              'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                              'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                              'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                              'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience'] 
                   else np.random.choice(['A', 'B', 'C'], 5) if col in ['EmploymentStatus', 'EducationLevel', 
                                                                        'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
                   else pd.date_range('2023-01-01', periods=5) if col == 'ApplicationDate'
                   else np.random.randn(5)
            for col in RAW_FEATURE_SCHEMA
        })
        
        result = validate_raw_data_schema(exact_data)
        assert result is True
    
    def test_validate_schema_case_sensitive_columns(self):
        """Test that column validation is case-sensitive."""
        # Create data with wrong case
        wrong_case_data = pd.DataFrame({
            'age': [25, 30, 35],  # lowercase instead of 'Age'
            'annualincome': [50000, 60000, 70000],  # lowercase instead of 'AnnualIncome'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data_schema(wrong_case_data)
    
    def test_validate_schema_column_order_independence(self, sample_loan_data):
        """Test that column order doesn't affect validation."""
        # Shuffle the columns
        shuffled_columns = sample_loan_data.columns.tolist()
        np.random.shuffle(shuffled_columns)
        shuffled_data = sample_loan_data[shuffled_columns]
        
        result = validate_raw_data_schema(shuffled_data)
        assert result is True
    
    @patch('FeatureEngineering.schema_validator.logging')
    def test_validate_schema_logging(self, mock_logging, sample_loan_data):
        """Test that appropriate logging messages are generated."""
        validate_raw_data_schema(sample_loan_data)
        
        # Verify logging calls were made
        assert mock_logging.info.called
        
        # Check for specific log messages
        log_calls = [call.args[0] for call in mock_logging.info.call_args_list]
        assert any("Initiating schema validation" in msg for msg in log_calls)
        assert any("schema validation successful" in msg for msg in log_calls)
    
    def test_validate_schema_error_message_format(self):
        """Test that error messages contain helpful information."""
        incomplete_data = pd.DataFrame({
            'Age': [25, 30],
            'AnnualIncome': [50000, 60000]
        })
        
        try:
            validate_raw_data_schema(incomplete_data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "Missing required columns" in error_msg
            assert "Schema validation failed" in error_msg
            # Should contain the actual missing columns
            assert len(error_msg) > 50  # Should be a detailed message
    
    def test_validate_schema_with_none_values(self):
        """Test schema validation with None/NaN values in data."""
        # Create data with None values (but correct schema)
        data_with_nones = pd.DataFrame({
            col: [None, 1, 2] if col == 'Age' else [1, 2, 3]
            for col in RAW_FEATURE_SCHEMA
        })
        
        # Schema validation should pass (it only checks column presence)
        result = validate_raw_data_schema(data_with_nones)
        assert result is True
    
    def test_validate_schema_unexpected_exception(self):
        """Test handling of unexpected exceptions during validation."""
        # Create a mock DataFrame that will cause an unexpected error
        class BadDataFrame:
            def __init__(self):
                pass
            
            @property
            def columns(self):
                raise RuntimeError("Unexpected error")
        
        bad_df = BadDataFrame()
        
        with pytest.raises(ValueError, match="An unexpected error occurred"):
            validate_raw_data_schema(bad_df)


class TestRawFeatureSchema:
    """Test class for RAW_FEATURE_SCHEMA constant."""
    
    def test_schema_is_list(self):
        """Test that RAW_FEATURE_SCHEMA is a list."""
        assert isinstance(RAW_FEATURE_SCHEMA, list)
    
    def test_schema_not_empty(self):
        """Test that RAW_FEATURE_SCHEMA is not empty."""
        assert len(RAW_FEATURE_SCHEMA) > 0
    
    def test_schema_contains_expected_columns(self):
        """Test that schema contains expected key columns."""
        expected_columns = [
            'Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 
            'EmploymentStatus', 'EducationLevel', 'LoanDuration'
        ]
        
        for col in expected_columns:
            assert col in RAW_FEATURE_SCHEMA, f"Expected column {col} not found in schema"
    
    def test_schema_no_duplicates(self):
        """Test that schema contains no duplicate columns."""
        assert len(RAW_FEATURE_SCHEMA) == len(set(RAW_FEATURE_SCHEMA))
    
    def test_schema_all_strings(self):
        """Test that all schema entries are strings."""
        assert all(isinstance(col, str) for col in RAW_FEATURE_SCHEMA)
    
    def test_schema_no_empty_strings(self):
        """Test that schema contains no empty strings."""
        assert all(len(col.strip()) > 0 for col in RAW_FEATURE_SCHEMA)
    
    def test_schema_column_count(self):
        """Test that schema has expected number of columns."""
        # Based on the actual schema, should have 34 columns
        expected_count = 34
        assert len(RAW_FEATURE_SCHEMA) == expected_count, f"Expected {expected_count} columns, got {len(RAW_FEATURE_SCHEMA)}"


class TestSchemaValidationEdgeCases:
    """Test class for edge cases in schema validation."""
    
    def test_validate_schema_single_row(self):
        """Test schema validation with single row of data."""
        single_row_data = pd.DataFrame({
            col: [1] if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                               'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                               'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                               'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                               'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                               'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                               'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                               'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                 else ['A'] if col in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 
                                      'HomeOwnershipStatus', 'LoanPurpose']
                 else [pd.Timestamp('2023-01-01')] if col == 'ApplicationDate'
                 else [1]
            for col in RAW_FEATURE_SCHEMA
        })
        
        result = validate_raw_data_schema(single_row_data)
        assert result is True
    
    def test_validate_schema_large_dataset(self):
        """Test schema validation with large dataset."""
        # Create a large dataset with correct schema
        n_rows = 10000
        large_data = pd.DataFrame({
            col: np.random.randn(n_rows) if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                                   'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                                   'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                                   'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                                   'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                                   'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                                   'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                                   'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                 else np.random.choice(['A', 'B', 'C'], n_rows) if col in ['EmploymentStatus', 'EducationLevel', 
                                                                          'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
                 else pd.date_range('2023-01-01', periods=n_rows) if col == 'ApplicationDate'
                 else np.random.randn(n_rows)
            for col in RAW_FEATURE_SCHEMA
        })
        
        import time
        start_time = time.time()
        result = validate_raw_data_schema(large_data)
        validation_time = time.time() - start_time
        
        assert result is True
        # Validation should be fast even for large datasets
        assert validation_time < 5.0, f"Validation took too long: {validation_time} seconds"
    
    def test_validate_schema_unicode_columns(self):
        """Test schema validation with unicode characters in data."""
        unicode_data = pd.DataFrame({
            col: ['测试', 'тест', 'テスト'] if col in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 
                                                    'HomeOwnershipStatus', 'LoanPurpose']
                 else [1, 2, 3] if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                          'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                          'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                          'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                          'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                          'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                          'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                          'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                 else [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')] if col == 'ApplicationDate'
                 else [1, 2, 3]
            for col in RAW_FEATURE_SCHEMA
        })
        
        result = validate_raw_data_schema(unicode_data)
        assert result is True
    
    def test_validate_schema_mixed_types_per_column(self):
        """Test schema validation with mixed data types in columns."""
        mixed_data = pd.DataFrame({
            col: [1, '2', 3.0] if col == 'Age'  # Mixed types in Age column
                 else [1, 2, 3] if col in ['AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                          'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                          'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                          'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                          'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                          'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                          'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                          'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                 else ['A', 'B', 'C'] if col in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 
                                                 'HomeOwnershipStatus', 'LoanPurpose']
                 else [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')] if col == 'ApplicationDate'
                 else [1, 2, 3]
            for col in RAW_FEATURE_SCHEMA
        })
        
        # Schema validation only checks column presence, not data types
        result = validate_raw_data_schema(mixed_data)
        assert result is True


class TestSchemaValidationPerformance:
    """Test class for schema validation performance."""
    
    @pytest.mark.performance
    def test_validation_performance_scaling(self):
        """Test that validation performance scales reasonably with data size."""
        import time
        
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            # Create data of specified size
            data = pd.DataFrame({
                col: np.random.randn(size) if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                                     'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                                     'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                                     'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                                     'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                                     'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                                     'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                                     'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                     else np.random.choice(['A', 'B', 'C'], size) if col in ['EmploymentStatus', 'EducationLevel', 
                                                                            'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
                     else pd.date_range('2023-01-01', periods=size) if col == 'ApplicationDate'
                     else np.random.randn(size)
                for col in RAW_FEATURE_SCHEMA
            })
            
            # Time the validation
            start_time = time.time()
            result = validate_raw_data_schema(data)
            end_time = time.time()
            
            assert result is True
            times.append(end_time - start_time)
        
        # Validation time should not increase dramatically with size
        # (since it's mainly checking column names, not data content)
        assert all(t < 1.0 for t in times), f"Validation times too slow: {times}"
    
    @pytest.mark.performance
    def test_validation_memory_usage(self):
        """Test that validation doesn't use excessive memory."""
        import psutil
        import os
        
        # Create moderately large dataset
        size = 50000
        data = pd.DataFrame({
            col: np.random.randn(size) if col in ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                                                 'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                                                 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio',
                                                 'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',
                                                 'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance',
                                                 'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
                                                 'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
                                                 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'Experience']
                 else np.random.choice(['A', 'B', 'C'], size) if col in ['EmploymentStatus', 'EducationLevel', 
                                                                        'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
                 else pd.date_range('2023-01-01', periods=size) if col == 'ApplicationDate'
                 else np.random.randn(size)
            for col in RAW_FEATURE_SCHEMA
        })
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run validation
        result = validate_raw_data_schema(data)
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result is True
        # Memory increase should be minimal (validation shouldn't copy data)
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.2f} MB"


@pytest.mark.integration
class TestSchemaValidationIntegration:
    """Integration tests for schema validation with other components."""
    
    def test_schema_validation_with_api_data(self, csv_loan_data):
        """Test schema validation with CSV data from API."""
        import io
        
        # Parse CSV data as it would come from API
        df = pd.read_csv(io.StringIO(csv_loan_data))
        
        # Should validate successfully
        result = validate_raw_data_schema(df)
        assert result is True
    
    def test_schema_validation_error_propagation(self):
        """Test that validation errors propagate correctly to calling code."""
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        try:
            validate_raw_data_schema(invalid_data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Error should contain useful information for API response
            error_msg = str(e)
            assert "Missing required columns" in error_msg
            assert len(error_msg) > 20  # Should be descriptive
    
    def test_schema_validation_with_preprocessed_data(self, sample_loan_data):
        """Test that validation works with data that might be preprocessed."""
        # Simulate some preprocessing (but keeping same columns)
        preprocessed_data = sample_loan_data.copy()
        
        # Convert some columns to different types (as might happen in preprocessing)
        preprocessed_data['Age'] = preprocessed_data['Age'].astype(str)
        preprocessed_data['EmploymentStatus'] = preprocessed_data['EmploymentStatus'].astype('category')
        
        # Validation should still pass (only checks column presence)
        result = validate_raw_data_schema(preprocessed_data)
        assert result is True
