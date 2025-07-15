import pandas as pd
import pickle
import os
import numpy as np

# Construct absolute paths to the data files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
X_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'X_train_processed.pkl')
Y_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'y_train.pkl')

def view_train_data_with_target(features_path, target_path):
    """
    Loads processed training data and target, combines them, and displays details,
    including the correlation of features with the target variable.
    """
    if not os.path.exists(features_path) or not os.path.exists(target_path):
        print(f"Error: Data file not found. Checked paths:\n- {features_path}\n- {target_path}")
        return

    try:
        # Load features and target data
        with open(features_path, 'rb') as f:
            X_train = pickle.load(f)
        with open(target_path, 'rb') as f:
            y_train = pickle.load(f)

        # Ensure X_train is a DataFrame
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        # Combine features and target into a single DataFrame
        combined_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

        # Set pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_rows', 150)

        print(f"--- Details for Combined Training Data ---")
        print(f"Features shape: {X_train.shape}")
        print(f"Target shape: {y_train.shape}")
        print(f"Combined shape: {combined_df.shape}")
        print("\n--- Data Head ---")
        print(combined_df.head())

        # Display data types
        print("\n--- Data Types ---")
        with pd.option_context('display.max_rows', None):
            print(combined_df.dtypes)

        # Check for missing values
        print("\n--- Missing Values Check ---")
        missing_values = combined_df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if missing_values.empty:
            print("No missing values found.")
        else:
            print(missing_values)

        # Calculate and display correlation with the target variable
        print(f"\n--- Correlation with Target Variable ({y_train.name}) ---")
        numeric_df = combined_df.select_dtypes(include=[np.number])
        correlation_with_target = numeric_df.corr()[y_train.name].sort_values(ascending=False)
        print(correlation_with_target)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    view_train_data_with_target(X_TRAIN_PATH, Y_TRAIN_PATH)
