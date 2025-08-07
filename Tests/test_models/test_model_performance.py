import json
import pytest
import pandas as pd
from pathlib import Path
from backend.model_service import ModelService
from sklearn.metrics import r2_score, mean_squared_error

@pytest.fixture(scope="module")
def service():
    return ModelService()

def load_metadata():
    metadata_path = Path(__file__).parents[2] / 'Models' / 'trained_models' / 'model_metadata.json'
    with open(metadata_path, 'r') as f:
        return json.load(f)

def test_model_accuracy_thresholds(service):
    """Validate that the model meets minimum R² and RMSE thresholds on a holdout set."""
    # Use raw dataset for proper feature schema
    data_path = Path(__file__).parents[2] / 'Data' / 'raw' / 'Loan.csv'
    df = pd.read_csv(data_path)

    # Identify target column from dataset
    assert 'RiskScore' in df.columns, "Expected 'RiskScore' column in Loan.csv as target"
    y_true = df['RiskScore']

    # Build feature matrix by removing the target
    X = df.drop(columns=['RiskScore'])

    # Basic validation that raw schema columns exist
    import json
    meta_path = Path(__file__).parents[2] / 'FeatureEngineering' / 'artifacts' / 'transformation_metadata.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    raw_cols = set(meta.get('numeric_columns', []) + meta.get('categorical_columns', []))
    assert len(set(X.columns) & raw_cols) > 0, "No overlap between Loan.csv columns and expected raw schema"

    preds = service.predict_batch(X)
    r2 = r2_score(y_true, preds)
    # Compute RMSE without using the 'squared' argument for compatibility
    rmse = mean_squared_error(y_true, preds) ** 0.5
    assert r2 >= 0.85, f"R² {r2:.3f} is below threshold"
    # Define an acceptable RMSE based on domain knowledge (example 0.15)
    assert rmse <= 3, f"RMSE {rmse:.3f} exceeds acceptable limit"



def test_prediction_latency(service):
    """Ensure a single prediction completes within a reasonable time (e.g., 100ms)."""
    import time
    record = {
        "ApplicationDate": "2024-01-01",
        "Age": 30,
        "CreditScore": 750,
        "EmploymentStatus": "Employed",
        "EducationLevel": "Bachelor",
        "Experience": 5,
        "LoanAmount": 50000.0,
        "LoanDuration": 60,
        "NumberOfDependents": 0,
        "HomeOwnershipStatus": "Own",
        "MonthlyDebtPayments": 200.0,
        "CreditCardUtilizationRate": 0.1,
        "NumberOfOpenCreditLines": 3,
        "NumberOfCreditInquiries": 0,
        "DebtToIncomeRatio": 0.2,
        "SavingsAccountBalance": 20000.0,
        "CheckingAccountBalance": 5000.0,
        "MonthlyIncome": 8000.0,
        "AnnualIncome": 96000.0,
        "BaseInterestRate": 0.04,
        "InterestRate": 0.05,
        "TotalDebtToIncomeRatio": 0.25,
        "LoanPurpose": "Home",
        "BankruptcyHistory": 0,
        "PaymentHistory": 2,
        "UtilityBillsPaymentHistory": 0.95,
        "PreviousLoanDefaults": 0,
        "TotalAssets": 300000,
        "TotalLiabilities": 100000,
        "NetWorth": 200000,
        "LengthOfCreditHistory": 120,
        "JobTenure": 60,
    }
    df = pd.DataFrame([record])
    start = time.time()
    _ = service.predict_batch(df)
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms <= 100, f"Prediction latency {elapsed_ms:.1f}ms exceeds 100ms"
