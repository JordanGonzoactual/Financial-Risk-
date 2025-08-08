import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of samples (smaller than the normal dataset to keep it quick)
num_samples = 1000

# Seed for reproducibility
np.random.seed(42)

# Helper functions to generate absurd values

def random_dates_future(size):
    # Far-future dates (e.g., year 2100+)
    start_date = datetime(2100, 1, 1)
    return [start_date + timedelta(days=int(i)) for i in np.random.randint(0, 36500, size=size)]

# Numeric generators with absurd ranges
rand_int = np.random.randint
rand_float = np.random.uniform

# Categorical with invalid choices
INVALID_EMPLOYMENT = np.array(['Alien', 'Time Traveler', 'Ghost', '', None], dtype=object)
INVALID_EDU = np.array(['Grade -1', 'Unknown', '???', '', None], dtype=object)
INVALID_MARITAL = np.array(['Complicated', 'Undefined', 'Parallel', '', None], dtype=object)
INVALID_HOME = np.array(['Spaceship', 'Cave', 'Unknown', '', None], dtype=object)
INVALID_PURPOSE = np.array(['Teleportation', 'Time Machine', 'Alchemy', '', None], dtype=object)

# Build absurd columns
ApplicationDate = random_dates_future(num_samples)
Age = rand_int(-10, 301, size=num_samples)  # negative to >300
Experience = rand_int(-20, 201, size=num_samples)  # negative or unrealistically high
EducationLevel = np.random.choice(INVALID_EDU, size=num_samples)
AnnualIncome = np.where(np.random.rand(num_samples) < 0.5,
                        -rand_int(1, 1_000_000, size=num_samples),  # negative income
                        rand_int(100_000_000, 1_000_000_000, size=num_samples))  # absurdly high
CreditScore = rand_int(-200, 1201, size=num_samples)  # <0 and >850
EmploymentStatus = np.random.choice(INVALID_EMPLOYMENT, size=num_samples)
Experience = rand_int(-50, 151, size=num_samples)

LoanAmount = np.where(np.random.rand(num_samples) < 0.5,
                      -rand_int(1_000, 1_000_000, size=num_samples),  # negative loan amount
                      rand_int(100_000_000, 2_000_000_000, size=num_samples))  # absurdly high
LoanDuration = rand_int(-120, 10001, size=num_samples)  # negative months to 10k months
MaritalStatus = np.random.choice(INVALID_MARITAL, size=num_samples)
NumberOfDependents = np.where(np.random.rand(num_samples) < 0.5,
                              -rand_int(1, 10, size=num_samples),  # negative dependents
                              rand_int(50, 1000, size=num_samples))  # unbelievable counts
HomeOwnershipStatus = np.random.choice(INVALID_HOME, size=num_samples)
MonthlyDebtPayments = np.where(np.random.rand(num_samples) < 0.5,
                               -rand_int(100, 50_000, size=num_samples),
                               rand_int(100_000, 10_000_000, size=num_samples))
CreditCardUtilizationRate = np.where(np.random.rand(num_samples) < 0.5,
                                     rand_float(-5.0, 0.0, size=num_samples),  # negative
                                     rand_float(1.2, 10.0, size=num_samples))  # >1000%
NumberOfOpenCreditLines = np.where(np.random.rand(num_samples) < 0.5,
                                   -rand_int(1, 20, size=num_samples),
                                   rand_int(50, 5000, size=num_samples))
NumberOfCreditInquiries = np.where(np.random.rand(num_samples) < 0.5,
                                   -rand_int(1, 20, size=num_samples),
                                   rand_int(50, 2000, size=num_samples))
DebtToIncomeRatio = np.where(np.random.rand(num_samples) < 0.5,
                             rand_float(-5.0, 0.0, size=num_samples),  # negative
                             rand_float(2.0, 10.0, size=num_samples))  # 200%-1000%
BankruptcyHistory = np.random.choice(np.array([-1, 0, 1, 2, 'yes', 'no'], dtype=object), size=num_samples)
LoanPurpose = np.random.choice(INVALID_PURPOSE, size=num_samples)
PreviousLoanDefaults = np.where(np.random.rand(num_samples) < 0.5,
                                -rand_int(1, 10, size=num_samples),
                                rand_int(20, 500, size=num_samples))
PaymentHistory = np.where(np.random.rand(num_samples) < 0.5,
                          -rand_int(1, 50, size=num_samples),
                          rand_int(101, 500, size=num_samples))  # percent-like >100
LengthOfCreditHistory = rand_int(-50, 1001, size=num_samples)
SavingsAccountBalance = -rand_int(1_000, 10_000_000, size=num_samples)  # negative balances
CheckingAccountBalance = -rand_int(1_000, 10_000_000, size=num_samples)
TotalAssets = -rand_int(1_000, 10_000_000, size=num_samples)  # negative assets
TotalLiabilities = -rand_int(1_000, 10_000_000, size=num_samples)  # negative liabilities (nonsense)
MonthlyIncome = np.where(np.random.rand(num_samples) < 0.5,
                         0,  # zero income to trigger div-by-zero/infinite ratios
                         -rand_int(1, 100_000, size=num_samples))  # negative
UtilityBillsPaymentHistory = np.where(np.random.rand(num_samples) < 0.5,
                                      rand_float(-2.0, 0.0, size=num_samples),
                                      rand_float(1.5, 3.0, size=num_samples))
JobTenure = rand_int(-10, 201, size=num_samples)

# Interest rate fields with absurd ranges
BaseInterestRate = rand_float(-0.5, 3.0, size=num_samples)
InterestRate = BaseInterestRate * rand_float(0.5, 2.5, size=num_samples) + rand_float(-0.5, 1.5, size=num_samples)

# Compute monthly payment with guard, then deliberately corrupt some values
r = InterestRate / 12.0
months = np.where(LoanDuration <= 0, 1, LoanDuration)  # avoid zero/negative in exponent
with np.errstate(all='ignore'):
    denom = 1 - np.power(1 + r, -months)
    MonthlyLoanPayment = (LoanAmount * r) / denom

# Deliberately inject NaNs, Infs, and negative payments
mask = np.random.rand(num_samples) < 0.3
MonthlyLoanPayment[mask] = np.nan
mask2 = (~mask) & (np.random.rand(num_samples) < 0.3)
MonthlyLoanPayment[mask2] = np.inf
mask3 = (~mask) & (~mask2)
MonthlyLoanPayment[mask3] = MonthlyLoanPayment[mask3] * np.where(np.random.rand(mask3.sum()) < 0.5, -10, 100)

# Total DTI using corrupted income (may yield inf/negatives)
with np.errstate(all='ignore'):
    TotalDebtToIncomeRatio = (MonthlyDebtPayments + np.nan_to_num(MonthlyLoanPayment, nan=0.0)) / MonthlyIncome

# Net worth without any min clamp, often extremely negative
NetWorth = TotalAssets - TotalLiabilities - rand_int(0, 10_000_000, size=num_samples)

# Assemble DataFrame preserving the expected schema
columns = {
    'ApplicationDate': ApplicationDate,
    'Age': Age,
    'AnnualIncome': AnnualIncome,
    'CreditScore': CreditScore,
    'EmploymentStatus': EmploymentStatus,
    'EducationLevel': EducationLevel,
    'Experience': Experience,
    'LoanAmount': LoanAmount,
    'LoanDuration': LoanDuration,
    'MaritalStatus': MaritalStatus,
    'NumberOfDependents': NumberOfDependents,
    'HomeOwnershipStatus': HomeOwnershipStatus,
    'MonthlyDebtPayments': MonthlyDebtPayments,
    'CreditCardUtilizationRate': CreditCardUtilizationRate,
    'NumberOfOpenCreditLines': NumberOfOpenCreditLines,
    'NumberOfCreditInquiries': NumberOfCreditInquiries,
    'DebtToIncomeRatio': DebtToIncomeRatio,
    'BankruptcyHistory': BankruptcyHistory,
    'LoanPurpose': LoanPurpose,
    'PreviousLoanDefaults': PreviousLoanDefaults,
    'PaymentHistory': PaymentHistory,
    'LengthOfCreditHistory': LengthOfCreditHistory,
    'SavingsAccountBalance': SavingsAccountBalance,
    'CheckingAccountBalance': CheckingAccountBalance,
    'TotalAssets': TotalAssets,
    'TotalLiabilities': TotalLiabilities,
    'MonthlyIncome': MonthlyIncome,
    'UtilityBillsPaymentHistory': UtilityBillsPaymentHistory,
    'JobTenure': JobTenure,
    'NetWorth': NetWorth,
    'BaseInterestRate': BaseInterestRate,
    'InterestRate': InterestRate,
    'MonthlyLoanPayment': MonthlyLoanPayment,
    'TotalDebtToIncomeRatio': TotalDebtToIncomeRatio,
}

df = pd.DataFrame(columns)

# Inject NaNs across columns randomly
for col in df.columns:
    mask_nan = np.random.rand(num_samples) < 0.1
    df.loc[mask_nan, col] = np.nan

# Save to CSV (no label columns)
output_name = 'absurd_synthetic_loan_data.csv'
df.to_csv(output_name, index=False)
print(f"Absurd synthetic data saved to '{output_name}' with {len(df)} rows and {len(df.columns)} columns.")

# Quick peek of extreme values
print(df.describe(include='all').transpose().head(10))
print("\nColumns:")
for c in df.columns:
    print(f"- {c}")
