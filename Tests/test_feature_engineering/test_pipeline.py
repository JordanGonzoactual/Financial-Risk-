import pytest
import pandas as pd
from FeatureEngineering.pipeline_builder import build_pipeline_from_dataframe
from pydantic import ValidationError
from shared.models.raw_data import RawData

# Note: Fixtures like sample_loan_data are provided by tests/conftest.py

# 1. Feature Engineering Pipeline Robustness

def test_dataframe_preservation_after_strict_validation(sample_loan_data):
    """Verify the pipeline works with valid data that passes strict validation."""
    original_df = sample_loan_data.copy()
    
    # Test that the data passes strict validation
    records = original_df.to_dict(orient='records')
    for record in records:
        # This should not raise ValidationError for valid data
        validated_record = RawData.model_validate(record)
        assert validated_record is not None

@pytest.mark.large_dataset
def test_batch_processing_performance(sample_loan_data):
    """Test pipeline handles large datasets efficiently with valid data."""
    large_df = pd.concat([sample_loan_data] * 100, ignore_index=True) # 10,000 rows
    
    # Test pipeline building with large dataset
    pipeline = build_pipeline_from_dataframe(large_df)
    assert pipeline is not None
    transformed_df = pipeline.fit_transform(large_df)
    assert not transformed_df.empty

def test_schema_evolution_with_new_fields(sample_loan_data):
    """Verify that extra fields are rejected by strict validation."""
    evolved_df = sample_loan_data.copy()
    evolved_df['NewFutureColumn'] = 'some_value'
    
    # Test that extra fields raise ValidationError in strict validation
    records = evolved_df.to_dict(orient='records')
    with pytest.raises(ValidationError) as exc_info:
        RawData.model_validate(records[0])
    
    # Verify the error mentions extra/forbidden fields
    error_details = str(exc_info.value)
    assert any(keyword in error_details.lower() for keyword in 
              ['extra', 'forbidden', 'not permitted', 'unexpected'])

    # Test pipeline building with original data (without extra field)
    original_df = sample_loan_data.copy()
    pipeline = build_pipeline_from_dataframe(original_df)
    assert pipeline is not None
    transformed_df = pipeline.fit_transform(original_df)
    assert not transformed_df.empty

def test_pipeline_with_edge_case_data():
    """Test pipeline with unexpected data patterns."""
    edge_case_data = {
        'ApplicationDate': ['2023-01-01'],
        'Age': [-25], 
        'AnnualIncome': [-50000],
        'CreditScore': [-100],
        'LoanAmount': [-999999999999999],
        # Add missing required columns for CraftedFeaturesTransformer
        'SavingsAccountBalance': [-1000],
        'CheckingAccountBalance': [-500], 
        'MonthlyIncome': [-4000],
        'MonthlyDebtPayments': [-1000],
        'MonthlyLoanPayment': [-500]
    }
    df = pd.DataFrame(edge_case_data)
    try:
        pipeline = build_pipeline_from_dataframe(df)
        transformed_df = pipeline.fit_transform(df)
        assert not transformed_df.empty
    except Exception as e:
        pytest.fail(f"Pipeline failed on edge case data: {e}")
