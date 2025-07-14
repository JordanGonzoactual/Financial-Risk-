import pandas as pd
import pickle
import os
from ydata_profiling import ProfileReport

# Define paths to the processed data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
X_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'X_train_processed.pkl')
Y_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'y_train.pkl')
REPORT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'EDA', 'processed_loan_analysis_report.html')

def generate_eda_report(features_path, target_path, output_path):
    """
    Loads processed training data, combines it with the target, and generates
    a detailed EDA report using ydata-profiling.
    """
    print("--- Starting EDA Report Generation ---")
    
    # Validate paths
    if not os.path.exists(features_path) or not os.path.exists(target_path):
        print(f"Error: Input data not found. Please run the feature engineering pipeline first.")
        print(f"Checked for: \n- {features_path}\n- {target_path}")
        return

    # Load the processed data
    print(f"Loading data from {features_path} and {target_path}...")
    with open(features_path, 'rb') as f:
        X_train = pickle.load(f)
    with open(target_path, 'rb') as f:
        y_train = pickle.load(f)

    # Combine features and target
    combined_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    print(f"Data loaded and combined. Shape: {combined_df.shape}")

    # Generate the report
    print("Generating ydata-profiling report...")
    profile = ProfileReport(
        combined_df, 
        title='Processed Loan Data Analysis',
        explorative=True,
        correlations={
            'auto': {'calculate': True},
            'pearson': {'calculate': True},
            'spearman': {'calculate': True}
        },
        interactions={'targets': ['RiskScore']}
    )

    # Save the report
    print(f"Saving report to {output_path}...")
    profile.to_file(output_path)
    print("--- EDA Report Generated Successfully ---")

if __name__ == "__main__":
    generate_eda_report(X_TRAIN_PATH, Y_TRAIN_PATH, REPORT_OUTPUT_PATH)
