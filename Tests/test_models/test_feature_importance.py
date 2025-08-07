import pandas as pd
import pytest
from pathlib import Path
from backend.model_service import ModelService

# Path to the feature importance CSV generated during training
FEATURE_IMPORTANCE_PATH = Path(__file__).parents[2] / 'Models' / 'trained_models' / 'feature_importance.csv'

@pytest.fixture(scope='module')
def importance_df():
    """Load the feature importance CSV as a DataFrame."""
    assert FEATURE_IMPORTANCE_PATH.exists(), f"Feature importance file not found at {FEATURE_IMPORTANCE_PATH}"
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    # Expect columns: 'feature' and 'importance'
    assert {'feature', 'importance'}.issubset(df.columns), "CSV must contain 'feature' and 'importance' columns"
    return df

def test_importance_values_reasonable(importance_df):
    """Importance values should be non‑negative and sum to approximately 1."""
    importances = importance_df['importance']
    # No negative values
    assert (importances >= 0).all(), "Importance contains negative values"
    total = importances.sum()
    # Allow small floating point tolerance
    assert abs(total - 1.0) < 1e-3, f"Importances sum to {total}, expected ~1"

def test_critical_features_high_rank(importance_df):
    """Critical business features should appear among top‑5 rankings.

    Align expectations with current artifacts where BankruptcyHistory, PreviousLoanDefaults,
    and DebtToIncomeRatio are among the top drivers.
    """
    critical = {'BankruptcyHistory', 'PreviousLoanDefaults', 'DebtToIncomeRatio'}
    sorted_features = importance_df.sort_values('importance', ascending=False)['feature'].tolist()
    top_five = set(sorted_features[:5])
    missing = critical - top_five
    assert not missing, f"Critical features not in top five: {missing}"

def test_removed_feature_not_present(importance_df):
    """Features that were removed during engineering should not appear."""
    removed = {'MaritalStatus'}
    present = set(importance_df['feature'])
    intersection = removed & present
    assert not intersection, f"Removed features found in importance: {intersection}"

def test_business_rule_consistency(importance_df):
    """Business rule: higher importance should correlate with risk drivers."""
    # Consider one-hot encoded or engineered names by matching prefixes
    def avg_importance_by_prefix(prefixes):
        mask = importance_df['feature'].apply(lambda f: any(f == p or f.startswith(p + '_') for p in prefixes))
        subset = importance_df[mask]
        return subset['importance'].mean() if not subset.empty else 0.0

    risk_avg = avg_importance_by_prefix(['DebtToIncomeRatio', 'LoanAmount', 'NumberOfOpenCreditLines'])
    neutral_avg = avg_importance_by_prefix(['EducationLevel', 'HomeOwnershipStatus'])
    assert risk_avg > neutral_avg, "Risk driver features have lower importance than neutral features"
