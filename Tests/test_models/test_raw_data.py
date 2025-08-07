"""Tests for RawData Pydantic model validation."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from shared.models.raw_data import (
    RawData,
    EmploymentStatus,
    EducationLevel,
    HomeOwnershipStatus,
    LoanPurpose
)

# Valid test data constants
VALID_EMPLOYMENT_STATUSES = ['Employed', 'Self-Employed', 'Unemployed']
VALID_EDUCATION_LEVELS = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
VALID_HOME_OWNERSHIP = ['Own', 'Rent', 'Mortgage', 'Other']
VALID_LOAN_PURPOSES = ['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other']

from Tests.conftest import get_valid_raw_data


class TestRawDataValidData:
    """Test that valid financial data passes validation successfully."""
    
    def test_valid_complete_data(self):
        """Test that a complete valid dataset creates a model instance successfully."""
        valid_data = get_valid_raw_data()
        model = RawData(**valid_data)
        
        # Verify model was created and fields are accessible
        assert model.age == 35
        assert model.credit_score == 720
        assert model.employment_status == "Employed"
        assert model.loan_amount == 250000.0
    
    @pytest.mark.parametrize("age", [18, 25, 65, 100])
    def test_valid_age_boundaries(self, age):
        """Test that valid age boundary values are accepted."""
        valid_data = get_valid_raw_data()
        valid_data["Age"] = age
        model = RawData(**valid_data)
        assert model.age == age
    
    @pytest.mark.parametrize("credit_score", [300, 500, 750, 850])
    def test_valid_credit_score_boundaries(self, credit_score):
        """Test that valid credit score boundary values are accepted."""
        valid_data = get_valid_raw_data()
        valid_data["CreditScore"] = credit_score
        model = RawData(**valid_data)
        assert model.credit_score == credit_score
    
    @pytest.mark.parametrize("employment_status", [
        "Employed",
        "Self-Employed", 
        "Unemployed"
    ])
    def test_valid_employment_status_values(self, employment_status):
        """Test that all valid employment status values are accepted."""
        valid_data = get_valid_raw_data()
        valid_data["EmploymentStatus"] = employment_status
        model = RawData(**valid_data)
        # With use_enum_values=True, the model returns string values
        assert model.employment_status == employment_status
    
    @pytest.mark.parametrize("education_level", [
        "High School",
        "Associate",
        "Bachelor",
        "Master",
        "Doctorate"
    ])
    def test_valid_education_level_values(self, education_level):
        """Test that all valid education level values are accepted."""
        valid_data = get_valid_raw_data()
        valid_data["EducationLevel"] = education_level
        model = RawData(**valid_data)
        # With use_enum_values=True, the model returns string values
        assert model.education_level == education_level


class TestRawDataInvalidTypes:
    """Test that invalid data types raise ValidationError as expected."""
    
    @pytest.mark.parametrize("field,invalid_value", [
        ("Age", "not-an-age"),
        ("Age", 25.5),  # Should be int, not float
        ("CreditScore", "not-a-score"),
        ("CreditScore", 720.5),  # Should be int, not float
        ("EmploymentStatus", 123),  # Should be string
        ("EducationLevel", 456),  # Should be string
        ("Experience", "not-experience"),
        ("LoanAmount", "not-amount"),
        ("LoanDuration", "not-duration"),
        ("NumberOfDependents", "not-number"),
        ("HomeOwnershipStatus", 789),  # Should be string
        ("MonthlyDebtPayments", "not-payments"),
        ("CreditCardUtilizationRate", "not-rate"),
        ("NumberOfOpenCreditLines", "not-lines"),
        ("NumberOfCreditInquiries", "not-inquiries"),
        ("DebtToIncomeRatio", "not-ratio"),
        ("LoanPurpose", 101112),  # Should be string
    ])
    def test_invalid_types_raise_validation_error(self, field, invalid_value):
        """Test that invalid data types raise ValidationError."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value
        
        with pytest.raises(ValidationError):
            RawData(**valid_data)


class TestRawDataOutOfRange:
    """Test that out-of-range values raise ValidationError as expected."""
    
    @pytest.mark.parametrize("field,invalid_value", [
        ("Age", 17),  # Below minimum (18)
        ("Age", 101),  # Above maximum (100)
        ("CreditScore", 299),  # Below minimum (300)
        ("CreditScore", 851),  # Above maximum (850)
        ("Experience", -1),  # Below minimum (0)
        ("LoanAmount", -1000),  # Below minimum (0)
        ("LoanDuration", 0),  # Below minimum (1)
        ("NumberOfDependents", -1),  # Below minimum (0)
        ("MonthlyDebtPayments", -500),  # Below minimum (0)
        ("CreditCardUtilizationRate", -0.1),  # Below minimum (0)
        ("CreditCardUtilizationRate", 1.1),  # Above maximum (1)
        ("NumberOfOpenCreditLines", -1),  # Below minimum (0)
        ("NumberOfCreditInquiries", -1),  # Below minimum (0)
        ("DebtToIncomeRatio", -0.1),  # Below minimum (0)
    ])
    def test_out_of_range_values_raise_validation_error(self, field, invalid_value):
        """Test that out-of-range values raise ValidationError."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value
        
        with pytest.raises(ValidationError):
            RawData(**valid_data)


class TestRawDataMissingFields:
    """Test that missing required fields raise ValidationError as expected."""
    
    def get_minimal_required_data(self):
        """Get minimal data with only some required fields."""
        return {
            "ApplicationDate": "2024-01-15",
            "Age": 35,
            "AnnualIncome": 75000.0,
            "CreditScore": 720,
            "EmploymentStatus": "Employed"
        }

    
    def test_missing_required_fields_raise_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        incomplete_data = self.get_minimal_required_data()
        
        with pytest.raises(ValidationError) as exc_info:
            RawData(**incomplete_data)
        
        # Verify the error mentions missing fields
        error_details = str(exc_info.value)
        assert 'field required' in error_details.lower() or 'missing' in error_details.lower()


class TestRawDataExtraFields:
    """Test that extra fields are rejected due to extra='forbid' configuration."""

    
    @pytest.mark.parametrize("extra_field,extra_value", [
        ("ExtraField", "ExtraValue"),
        ("UnknownProperty", 123),
        ("InvalidField", [1, 2, 3]),
        
    ])
    def test_extra_fields_raise_validation_error(self, extra_field, extra_value):
        """Test that extra fields are rejected due to extra='forbid' configuration."""
        valid_data = get_valid_raw_data()
        valid_data[extra_field] = extra_value
        
        with pytest.raises(ValidationError) as exc_info:
            RawData(**valid_data)
        
        # Verify the error mentions extra/forbidden fields
        error_details = str(exc_info.value)
        assert any(keyword in error_details.lower() for keyword in 
                  ['extra', 'forbidden', 'not permitted', 'unexpected'])


class TestRawDataEnumValidation:
    """Tests for enum field validation in RawData model."""


    @pytest.mark.parametrize("field,invalid_value", [
        ("EmploymentStatus", "retired"),  # Invalid option
        ("HomeOwnershipStatus", "lease"),  # Invalid option
        ("LoanPurpose", "personal"),  # Invalid option
        ("EducationLevel", "phd"),  # Invalid option
    ])
    def test_invalid_enum_values_raise_validation_error(self, field, invalid_value):
        """Test that invalid enum values raise ValidationError."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value

        with pytest.raises(ValidationError):
            RawData(**valid_data)





    @pytest.mark.parametrize("field,invalid_value", [
        ("InterestRate", -0.1),  # Below minimum (0)
        ("InterestRate", 1.1),  # Above maximum (1)
        ("InterestRate", "5%"),  # Invalid type
    ])
    def test_invalid_interest_rate_values(self, field, invalid_value):
        """Test that invalid interest rate values raise ValidationError."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value

        with pytest.raises(ValidationError):
            RawData(**valid_data)


class TestRawDataInterestRate:
    """Tests for interest rate field validation in RawData model."""

    @pytest.mark.parametrize("field,invalid_value", [
        ("InterestRate", -0.1),  # Below minimum (0)
        ("InterestRate", 1.1),  # Above maximum (1)
        ("InterestRate", "5%"),  # Invalid type
    ])
    def test_invalid_interest_rate_values(self, field, invalid_value):
        """Test that invalid interest rate values raise ValidationError."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value

        with pytest.raises(ValidationError):
            RawData(**valid_data)


class TestRawDataNewFields:
    """Tests for new fields in RawData model."""

    @pytest.mark.parametrize("field,invalid_value", [
        ("TotalAssets", -1000),  # Below minimum (0)
        ("TotalLiabilities", -500),  # Below minimum (0)
        ("LengthOfCreditHistory", -12),  # Below minimum (0)
        ("JobTenure", -6),  # Below minimum (0)
    ])
    def test_new_fields_validation(self, field, invalid_value):
        """Test that new financial fields validate correctly."""
        valid_data = get_valid_raw_data()
        valid_data[field] = invalid_value

        with pytest.raises(ValidationError):
            RawData(**valid_data)
