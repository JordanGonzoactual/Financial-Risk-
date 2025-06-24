import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('Data/Loan.csv')

# Convert ApplicationDate to datetime format
df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])

# Create a detailed report object
profile = ProfileReport(
    df, 
    title='Loan Dataset Analysis', 
    explorative=True, 
    correlations={
        'auto': {'calculate': True},
        'pearson': {'calculate': True},
        'spearman': {'calculate': True}
    },
    interactions={'targets': ['LoanApproved']}
)

# Generate and save the HTML report
profile.to_file('loan_analysis_report.html')

print("EDA report generated successfully as loan_analysis_report.html")
