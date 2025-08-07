import pytest
import pandas as pd
from backend.model_service import ModelService
from shared.models.raw_data import RawData

@pytest.fixture(scope='module')
def service():
    return ModelService()

def base_record():
    """Return a baseline valid record used for business logic variations."""
    return {
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
        "MonthlyLoanPayment": 1200.0,
    }

def predict(record, service):
    df = pd.DataFrame([record])
    # Validate schema via RawData model (will raise if invalid)
    RawData(**record)
    return service.predict_batch(df).iloc[0]

def test_perfect_credit_low_risk(service):
    """A perfect credit applicant should receive a low risk score."""
    rec = base_record()
    rec.update({
        "CreditScore": 850,
        "BankruptcyHistory": 0,
        "DebtToIncomeRatio": 0.1,
        "LoanAmount": 20000.0,
    })
    risk = predict(rec, service)
    assert 0.0 <= risk <= 35.0, f"Expected low risk, got {risk}"

def test_bankruptcy_high_risk(service):
    """An applicant with bankruptcy history should receive a high risk score."""
    rec = base_record()
    rec.update({
        "CreditScore": 400,
        "BankruptcyHistory": 1,
        "DebtToIncomeRatio": 0.9,
        "LoanAmount": 200000.0,
    })
    risk = predict(rec, service)
    assert risk >= 50.0, f"Expected high risk, got {risk}"

def test_debt_to_income_impact(service):
    """Higher debt-to-income ratio should increase risk score, all else equal."""
    rec_low = base_record()
    rec_low["DebtToIncomeRatio"] = 0.2
    rec_high = base_record()
    rec_high["DebtToIncomeRatio"] = 0.8
    risk_low = predict(rec_low, service)
    risk_high = predict(rec_high, service)
    assert risk_high > risk_low, "Risk should increase with higher DTI"

def test_credit_score_impact(service):
    """Higher credit score should decrease risk score, all else equal."""
    rec_low = base_record()
    rec_low["CreditScore"] = 400
    rec_high = base_record()
    rec_high["CreditScore"] = 800
    risk_low = predict(rec_low, service)
    risk_high = predict(rec_high, service)
    assert risk_high < risk_low, "Risk should decrease with higher credit score"

def test_income_level_impact(service):
    """Higher income should generally lower risk, holding other factors constant."""
    rec_low = base_record()
    rec_low["MonthlyIncome"] = 3000.0
    rec_low["AnnualIncome"] = 36000.0
    rec_high = base_record()
    rec_high["MonthlyIncome"] = 12000.0
    rec_high["AnnualIncome"] = 144000.0
    risk_low = predict(rec_low, service)
    risk_high = predict(rec_high, service)
    assert risk_high < risk_low, "Higher income should reduce risk"

def test_risk_score_interpretability(service):
    """Risk scores should be monotonic: higher score indicates higher risk."""
    recs = []
    # Create a gradient of risk by varying a single factor
    for dti in [10, 20, 30, 40, 50]:
        rec = base_record()
        rec["DebtToIncomeRatio"] = dti
        recs.append(rec)
    preds = [predict(r, service) for r in recs]
    # Ensure the list is non‑decreasing
    for earlier, later in zip(preds, preds[1:]):
        assert later >= earlier, "Risk scores should not decrease as DTI increases"
