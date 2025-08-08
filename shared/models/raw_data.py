from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Keep only these string enums for validation
class EmploymentStatus(str, Enum):
    EMPLOYED = "Employed"
    SELF_EMPLOYED = "Self-Employed"
    UNEMPLOYED = "Unemployed"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        value = value.lower().replace("-", "_")
        for member in cls:
            if member.value.lower().replace("-", "_") == value:
                return member
        return None

class HomeOwnershipStatus(str, Enum):
    OWN = "Own"
    RENT = "Rent"
    MORTGAGE = "Mortgage"
    OTHER = "Other"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        value = value.lower().replace("-", "_")
        for member in cls:
            if member.value.lower().replace("-", "_") == value:
                return member
        return None

class LoanPurpose(str, Enum):
    HOME = "Home"
    AUTO = "Auto"
    EDUCATION = "Education"
    DEBT_CONSOLIDATION = "Debt Consolidation"
    OTHER = "Other"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        value = value.lower().replace("-", "_")
        for member in cls:
            if member.value.lower().replace("-", "_") == value:
                return member
        return None

class EducationLevel(str, Enum):
    HIGH_SCHOOL = "High School"
    ASSOCIATE = "Associate"
    BACHELOR = "Bachelor"
    MASTER = "Master"
    DOCTORATE = "Doctorate"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        value = str(value).lower().strip()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None


class MaritalStatus(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"
    WIDOWED = "Widowed"

    @classmethod
    def _missing_(cls, value):
        value = str(value).lower().strip()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None

class RawData(BaseModel):
    """
    Pydantic model for raw financial data with strict validation.
    This model enforces proper data types, ranges, and business rules
    """
    application_date: str = Field(..., alias="ApplicationDate", description="Date of loan application")
    age: int = Field(..., alias="Age", ge=18, le=100, description="Applicant age in years")
    credit_score: int = Field(..., alias="CreditScore", ge=300, le=850, description="FICO credit score")
    employment_status: EmploymentStatus = Field(
        ..., 
        alias="EmploymentStatus",
        description="Current employment status"
    )
    education_level: EducationLevel = Field(
        ..., 
        alias="EducationLevel",
        description="Education level: High School, Associate, Bachelor, Master, Doctorate"
    )
    marital_status: Optional[MaritalStatus] = Field(
        None,
        alias="MaritalStatus",
        description="Marital status of the applicant"
    )
    experience: int = Field(..., alias="Experience", ge=0, description="Years of work experience")
    loan_amount: float = Field(..., alias="LoanAmount", gt=0, description="Requested loan amount")
    loan_duration: int = Field(..., alias="LoanDuration", gt=0, description="Loan term in months")
    number_of_dependents: int = Field(..., alias="NumberOfDependents", ge=0, description="Number of dependents")
    home_ownership_status: HomeOwnershipStatus = Field(
        ..., 
        alias="HomeOwnershipStatus",
        description="Current home ownership status"
    )
    monthly_debt_payments: float = Field(..., alias="MonthlyDebtPayments", ge=0, description="Total monthly debt payments")
    credit_card_utilization_rate: float = Field(..., alias="CreditCardUtilizationRate", ge=0, le=1, description="Credit card utilization ratio")
    number_of_open_credit_lines: int = Field(..., alias="NumberOfOpenCreditLines", ge=0, description="Total open credit lines")
    number_of_credit_inquiries: int = Field(..., alias="NumberOfCreditInquiries", ge=0, description="Recent credit inquiries")
    debt_to_income_ratio: float = Field(..., alias="DebtToIncomeRatio", ge=0, description="Debt-to-income ratio")
    savings_account_balance: float = Field(ge=0, alias="SavingsAccountBalance", description="Savings account balance")
    checking_account_balance: float = Field(ge=0, alias="CheckingAccountBalance", description="Checking account balance")
    monthly_income: float = Field(gt=0, alias="MonthlyIncome", description="Gross monthly income")
    annual_income: float = Field(gt=0, alias="AnnualIncome", description="Gross annual income")
    monthly_loan_payment: float = Field(ge=0, alias="MonthlyLoanPayment", description="Estimated monthly loan payment")
    loan_purpose: LoanPurpose = Field(
        ..., 
        alias="LoanPurpose",
        description="Purpose of the loan"
    )
    bankruptcy_history: int = Field(..., alias="BankruptcyHistory", ge=0, le=1, description="1=Has bankruptcy history, 0=No bankruptcy")
    payment_history: int = Field(..., alias="PaymentHistory", description="")
    utility_bills_payment_history: float = Field(..., alias="UtilityBillsPaymentHistory", ge=0, le=1, description="On-time payment percentage")
    previous_loan_defaults: int = Field(..., alias="PreviousLoanDefaults", ge=0, le=1, description="1=Has previous defaults, 0=No defaults")
    base_interest_rate: float = Field(
        ...,
        alias="BaseInterestRate",
        ge=0,
        le=1,
        description="Base interest rate before adjustments (0 to 1)"
    )
    interest_rate: float = Field(
        ..., 
        alias="InterestRate",
        ge=0,
        le=1,
        description="The interest rate for the loan (0 to 1)"
    )
    total_debt_to_income_ratio: float = Field(ge=0,
        alias="TotalDebtToIncomeRatio",
        description="Total debt to income ratio including new loan"
    )
    total_assets: int = Field(..., alias="TotalAssets", ge=0, description="Total assets value")
    total_liabilities: int = Field(..., alias="TotalLiabilities", ge=0, description="Total liabilities value")
    net_worth: int = Field(..., alias="NetWorth", description="Net worth (assets - liabilities)")
    length_of_credit_history: int = Field(..., alias="LengthOfCreditHistory", ge=0, description="Months of credit history")
    job_tenure: int = Field(..., alias="JobTenure", ge=0, description="Months at current job")

    class Config:
        use_enum_values = True
        extra = 'forbid'
