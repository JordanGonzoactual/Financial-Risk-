import pytest
import pandas as pd
import numpy as np
from backend.model_service import ModelService
from shared.models.raw_data import RawData
from .utils import get_valid_record

@pytest.fixture(scope='module')
def service():
    return ModelService()

def generate_record(overrides=None):
    """Return a baseline valid record, optionally overridden, using shared utility."""
    return get_valid_record(overrides)

def test_min_max_valid_values(service):
    """Test model with each feature at its documented min and max values."""
    # Example min/max ranges – these would normally be derived from data documentation.
    min_record = generate_record({
        "Age": 18,
        "CreditScore": 300,
        "LoanAmount": 1000.0,
        "LoanDuration": 12,
        "DebtToIncomeRatio": 0.0,
        "CreditCardUtilizationRate": 0.0,
        "NumberOfOpenCreditLines": 0,
        "NumberOfCreditInquiries": 0,
        "BankruptcyHistory": 0,
    })
    max_record = generate_record({
        "Age": 100,
        "CreditScore": 850,
        "LoanAmount": 1_000_000.0,
        "LoanDuration": 480,
        "DebtToIncomeRatio": 1.0,
        "CreditCardUtilizationRate": 1.0,
        "NumberOfOpenCreditLines": 20,
        "NumberOfCreditInquiries": 10,
        "BankruptcyHistory": 1,
    })
    df = pd.DataFrame([min_record, max_record])
    preds = service.predict_batch(df)
    assert len(preds) == 2
    # Predictions should be within the 0‑1 risk score range
    assert all(0.0 <= p <= 100.0 for p in preds)

def test_data_quality_issues(service):
    """Model should handle NaNs and outliers without crashing."""
    record = get_valid_record()
    # Introduce NaNs and extreme outliers
    record['Age'] = np.nan
    record['CreditScore'] = 2000  # extreme outlier
    df = pd.DataFrame([record])
    # The preprocessing pipeline is expected to impute or drop NaNs; we just ensure no exception
    preds = service.predict_batch(df)
    assert len(preds) == 1
    assert 0.0 <= preds.iloc[0] <= 100.0

def test_small_input_perturbations(service):
    """Predictions should be stable under tiny input changes (numerical robustness)."""
    base = get_valid_record()
    df_base = pd.DataFrame([base])
    pred_base = service.predict_batch(df_base).iloc[0]
    # Perturb each numeric field by a tiny epsilon
    eps = 1e-6
    for key, value in base.items():
        if isinstance(value, (int, float)):
            perturbed = base.copy()
            perturbed[key] = value + eps
            df_pert = pd.DataFrame([perturbed])
            pred_pert = service.predict_batch(df_pert).iloc[0]
            # Allow a very small difference due to floating point noise
            assert abs(pred_base - pred_pert) < 1e-4, f"Prediction changed for {key}"

def test_impossible_combinations(service):
    """Model should still produce a risk score for logically impossible combos."""
    record = get_valid_record({
        "Income": 1_000_000,  # extremely high income
        "BankruptcyHistory": 1,  # but has bankruptcy
        "LoanAmount": 500_000,
        "DebtToIncomeRatio": 0.9,
    })
    df = pd.DataFrame([record])
    preds = service.predict_batch(df)
    assert len(preds) == 1
    assert 0.0 <= preds.iloc[0] <= 100.0

def test_completely_invalid_input(service):
    """When input is entirely invalid, the service should raise a clear error."""
    # Create a DataFrame with unrelated columns
    df = pd.DataFrame([{"foo": "bar", "baz": 123}])
    with pytest.raises(Exception):
        service.predict_batch(df)
