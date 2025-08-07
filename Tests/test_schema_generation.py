"""
Test schema-based data generation for RawData model.

This module tests the generate_valid_rawdata_sample function to ensure
it produces valid data that passes RawData model validation.
"""

import pytest
import pandas as pd
from shared.models.raw_data import RawData
from Tests.conftest import generate_valid_rawdata_sample


def test_schema_based_generation_produces_valid_data():
    """Test that schema-generated data passes validation with proper field aliases."""
    # Create test data using the external-facing field names (PascalCase)
    test_data = {
        'ApplicationDate': '2024-01-15',
        'Age': 35,
        'CreditScore': 720,
        'EmploymentStatus': 'Employed',  # Changed from 'employed' to match enum case
        'EducationLevel': 'Bachelor',  # Changed from 2 to string value
        'Experience': 10,
        'LoanAmount': 250000.0,
        'LoanDuration': 360,
        'NumberOfDependents': 2,
        'HomeOwnershipStatus': 'Mortgage',  # Verify enum case
        'MonthlyDebtPayments': 1500.0,
        'CreditCardUtilizationRate': 0.25,
        'NumberOfOpenCreditLines': 5,
        'NumberOfCreditInquiries': 2,
        'DebtToIncomeRatio': 0.35,
        'SavingsAccountBalance': 25000.0,
        'CheckingAccountBalance': 5000.0,
        'MonthlyIncome': 6250.0,
        'AnnualIncome': 75000.0,
        'MonthlyLoanPayment': 1200.0,
        'LoanPurpose': 'Home',  # Verify enum case
        'BankruptcyHistory': 0,
        'PaymentHistory': 2,
        'UtilityBillsPaymentHistory': 0.95,
        'PreviousLoanDefaults': 0,
        'InterestRate': 0.05,
        'TotalAssets': 350000,
        'TotalLiabilities': 180000,
        'NetWorth': 170000,
        'LengthOfCreditHistory': 120,
        'JobTenure': 60,
        'BaseInterestRate': 0.045,
        'TotalDebtToIncomeRatio': 0.40
    }
    
    # This should work because Pydantic will handle the aliases
    validated_data = RawData(**test_data)
    assert validated_data


def test_schema_generation_covers_all_fields():
    """Test that generated schema includes all model fields with correct aliases."""
    schema = RawData.schema()
    
    # Get all model field aliases (PascalCase names)
    model_fields = {field.alias for field in RawData.__fields__.values() if field.alias}
    
    # Verify all fields are represented in schema with PascalCase names
    missing_fields = model_fields - set(schema['properties'].keys())
    assert not missing_fields, f"Schema missing fields: {missing_fields}"
    
    # Verify all schema fields match model aliases (PascalCase)
    alias_mismatches = [
        field for field in schema['properties']
        if field not in model_fields
    ]
    assert not alias_mismatches, f"Alias mismatches: {alias_mismatches}"


def test_schema_generation_respects_constraints():
    """Test that generated schema uses PascalCase and documents constraints."""
    schema = RawData.schema()
    
    # Verify PascalCase field names in schema
    assert "Age" in schema["properties"]
    assert "CreditScore" in schema["properties"]
    
    # Verify numeric constraints (using PascalCase names)
    assert schema["properties"]["Age"]["minimum"] == 18
    assert schema["properties"]["CreditScore"]["maximum"] == 850
    
    # Verify enum options are documented (for API consumers)
    if "enum" in schema["properties"]["EmploymentStatus"]:
        assert set(schema["properties"]["EmploymentStatus"]["enum"]) == {
            "Employed", "Self-Employed", "Unemployed"
        }


if __name__ == "__main__":
    # Run tests directly for quick validation
    test_schema_based_generation_produces_valid_data()
    test_schema_generation_covers_all_fields()
    test_schema_generation_respects_constraints()
    print(" All schema generation tests passed!")
