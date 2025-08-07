import pytest
from pydantic import ValidationError
from shared.models.raw_data import RawData

# Note: The validation_data_factory fixture is provided by tests/conftest.py

class TestValidationDataFactory:
    """Tests the functionality and utility of the validation_data_factory fixture."""

    def test_factory_for_missing_values(self, validation_data_factory):
        """Verify the factory correctly generates data with missing values that raise ValidationError."""
        df = validation_data_factory(issue_type='missing_values', field='Age', num_rows=10)
        records = df.to_dict(orient='records')
        
        # Missing values should raise ValidationError in strict validation
        with pytest.raises(ValidationError):
            RawData.model_validate(records[0])
        
        assert df['Age'].isnull().all()

    def test_factory_for_wrong_type(self, validation_data_factory):
        """Verify the factory correctly generates data with type errors that raise ValidationError."""
        df = validation_data_factory(issue_type='wrong_type', field='CreditScore', value='bad-score')
        records = df.to_dict(orient='records')
        
        # Wrong types should raise ValidationError in strict validation
        with pytest.raises(ValidationError) as exc_info:
            RawData.model_validate(records[0])
        
        # Verify the error is related to CreditScore field
        error_details = str(exc_info.value)
        assert 'creditscore' in error_details.lower() or 'CreditScore' in error_details

    def test_factory_for_out_of_range(self, validation_data_factory):
        """Verify the factory correctly generates out-of-range data that raises ValidationError."""
        df = validation_data_factory(issue_type='out_of_range', field='Age', value=-50)
        records = df.to_dict(orient='records')
        
        # Out of range values should raise ValidationError in strict validation
        with pytest.raises(ValidationError) as exc_info:
            RawData.model_validate(records[0])
        
        # Verify the error mentions range constraint
        error_details = str(exc_info.value)
        assert any(constraint in error_details.lower() for constraint in 
                  ['greater', 'less', 'range', 'minimum', 'maximum'])
        
        assert (df['Age'] == -50).all()
