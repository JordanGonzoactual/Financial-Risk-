"""Utility functions for test modules.

Provides a single source of truth for a valid raw data record used across
multiple test files. This helps keep the test suite DRY and ensures that any
future changes to the baseline record are reflected everywhere.
"""

from typing import Dict, Any

def get_valid_record(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a baseline valid record for the loan risk model.

    Parameters
    ----------
    overrides: dict | None
        Optional dictionary of fields to override in the baseline record.

    Returns
    -------
    dict
        A dictionary representing a valid input record.
    """
    base = {
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
    if overrides:
        base.update(overrides)
    return base
