import pytest
import pandas as pd
from backend.model_service import ModelService
from shared.models.raw_data import RawData

# Helper to create a DataFrame from a dict
def make_df(record: dict) -> pd.DataFrame:
    return pd.DataFrame([record])

@pytest.fixture(scope="module")
def service():
    return ModelService()

def test_prediction_low_risk(service):
    """A good applicant should receive a low risk score (close to 0)."""
    record = {
        "ApplicationDate": "2024-01-01",
        "Age": 30,
        "CreditScore": 800,
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
        "MonthlyLoanPayment": 1200.0,
    }
    df = make_df(record)
    # Validate input schema first
    validated = RawData(**record)
    preds = service.predict_batch(df)
    assert len(preds) == 1
    risk = preds.iloc[0]
    assert 0.0 <= risk <= 35.0, f"Low risk applicant got {risk}"

def test_prediction_high_risk(service):
    """A bad applicant should receive a high risk score (close to 1)."""
    record = {
        "ApplicationDate": "2024-01-01",
        "Age": 45,
        "CreditScore": 350,
        "EmploymentStatus": "Unemployed",
        "EducationLevel": "High School",
        "Experience": 0,
        "LoanAmount": 200000.0,
        "LoanDuration": 360,
        "NumberOfDependents": 4,
        "HomeOwnershipStatus": "Rent",
        "MonthlyDebtPayments": 3000.0,
        "CreditCardUtilizationRate": 0.9,
        "NumberOfOpenCreditLines": 1,
        "NumberOfCreditInquiries": 5,
        "DebtToIncomeRatio": 0.9,
        "SavingsAccountBalance": 1000.0,
        "CheckingAccountBalance": 500.0,
        "MonthlyIncome": 1500.0,
        "AnnualIncome": 18000.0,
        "BaseInterestRate": 0.08,
        "InterestRate": 0.09,
        "TotalDebtToIncomeRatio": 0.95,
        "LoanPurpose": "Debt Consolidation",
        "BankruptcyHistory": 1,
        "PaymentHistory": 0,
        "UtilityBillsPaymentHistory": 0.2,
        "PreviousLoanDefaults": 1,
        "TotalAssets": 5000,
        "TotalLiabilities": 250000,
        "NetWorth": -245000,
        "LengthOfCreditHistory": 12,
        "JobTenure": 0,
        "MonthlyLoanPayment": 2500.0,
    }
    df = make_df(record)
    validated = RawData(**record)
    preds = service.predict_batch(df)
    risk = preds.iloc[0]
    assert 60.0 <= risk <= 100.0, f"High risk applicant got {risk}"

def test_prediction_range_and_consistency(service):
    """Predictions should be within 0‑1 and deterministic for same input."""
    record = {
        "ApplicationDate": "2024-01-01",
        "Age": 40,
        "CreditScore": 600,
        "EmploymentStatus": "Self-Employed",
        "EducationLevel": "Associate",
        "Experience": 10,
        "LoanAmount": 100000.0,
        "LoanDuration": 120,
        "NumberOfDependents": 2,
        "HomeOwnershipStatus": "Mortgage",
        "MonthlyDebtPayments": 800.0,
        "CreditCardUtilizationRate": 0.5,
        "NumberOfOpenCreditLines": 4,
        "NumberOfCreditInquiries": 2,
        "DebtToIncomeRatio": 0.4,
        "SavingsAccountBalance": 15000.0,
        "CheckingAccountBalance": 3000.0,
        "MonthlyIncome": 5000.0,
        "AnnualIncome": 60000.0,
        "BaseInterestRate": 0.05,
        "InterestRate": 0.06,
        "TotalDebtToIncomeRatio": 0.45,
        "LoanPurpose": "Auto",
        "BankruptcyHistory": 0,
        "PaymentHistory": 1,
        "UtilityBillsPaymentHistory": 0.8,
        "PreviousLoanDefaults": 0,
        "TotalAssets": 120000,
        "TotalLiabilities": 80000,
        "NetWorth": 40000,
        "LengthOfCreditHistory": 60,
        "JobTenure": 24,
        "MonthlyLoanPayment": 1000.0,
    }
    df = make_df(record)
    preds1 = service.predict_batch(df)
    preds2 = service.predict_batch(df)
    risk = preds1.iloc[0]
    assert 0.0 <= risk <= 100.0
    assert preds1.equals(preds2), "Predictions should be deterministic"

def test_batch_prediction(service):
    """Batch prediction should return a series of correct length."""
    records = []
    for i in range(5):
        rec = {
            "ApplicationDate": "2024-01-01",
            "Age": 30 + i,
            "CreditScore": 750 - i*10,
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
            "MonthlyLoanPayment": 1200.0,
        }
        records.append(rec)
    df = pd.DataFrame(records)
    preds = service.predict_batch(df)
    assert len(preds) == 5
    assert all(0.0 <= r <= 100.0 for r in preds)
