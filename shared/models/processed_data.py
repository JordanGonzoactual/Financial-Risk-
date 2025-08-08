from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

class ProcessedData(BaseModel):
    """
    Processed version of RawData with encoded numerical features.
    Contains all columns from loan.csv except MaritalStatus, LoanApproved, and RiskScore.
    """
    
    # Application metadata
    application_date: str = Field(..., alias="ApplicationDate")
    
    # Applicant information
    age: int = Field(..., alias="Age", ge=18, le=100)
    credit_score: int = Field(..., alias="CreditScore", ge=300, le=850)
    experience: int = Field(..., alias="Experience", ge=0)
    job_tenure: int = Field(..., alias="JobTenure", ge=0)
    
    # Encoded categoricals (direct replacements of original columns)
    employment_status: int = Field(..., alias="EmploymentStatus", ge=0)
    education_level: int = Field(..., alias="EducationLevel", ge=0, le=4)
    home_ownership_status: int = Field(..., alias="HomeOwnershipStatus", ge=0)
    loan_purpose: int = Field(..., alias="LoanPurpose", ge=0)
    
    # Financial information
    loan_amount: float = Field(..., alias="LoanAmount", gt=0)
    loan_duration: int = Field(..., alias="LoanDuration", gt=0)
    monthly_income: float = Field(..., alias="MonthlyIncome", gt=0)
    annual_income: float = Field(..., alias="AnnualIncome", gt=0)
    monthly_debt_payments: float = Field(..., alias="MonthlyDebtPayments", ge=0)
    credit_card_utilization_rate: float = Field(..., alias="CreditCardUtilizationRate", ge=0, le=1)
    debt_to_income_ratio: float = Field(..., alias="DebtToIncomeRatio", ge=0)
    interest_rate: float = Field(..., alias="InterestRate", ge=0, le=1)
    
    # Credit history
    number_of_dependents: int = Field(..., alias="NumberOfDependents", ge=0)
    number_of_open_credit_lines: int = Field(..., alias="NumberOfOpenCreditLines", ge=0)
    number_of_credit_inquiries: int = Field(..., alias="NumberOfCreditInquiries", ge=0)
    length_of_credit_history: int = Field(..., alias="LengthOfCreditHistory", ge=0)
    payment_history: int = Field(..., alias="PaymentHistory", ge=0, le=3)
    bankruptcy_history: int = Field(..., alias="BankruptcyHistory", ge=0, le=1)
    previous_loan_defaults: int = Field(..., alias="PreviousLoanDefaults", ge=0, le=1)
    utility_bills_payment_history: float = Field(..., alias="UtilityBillsPaymentHistory", ge=0, le=1)
    
    # Assets
    savings_account_balance: float = Field(..., alias="SavingsAccountBalance", ge=0)
    checking_account_balance: float = Field(..., alias="CheckingAccountBalance", ge=0)
    total_assets: int = Field(..., alias="TotalAssets", ge=0)
    total_liabilities: int = Field(..., alias="TotalLiabilities", ge=0)
    net_worth: int = Field(..., alias="NetWorth")
    
    # Loan details
    monthly_loan_payment: float = Field(..., alias="MonthlyLoanPayment", ge=0)
    
    class Config:
        arbitrary_types_allowed = True
