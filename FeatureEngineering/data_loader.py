import os
import pickle
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(processed_data_path: str) -> tuple:
    """
    Loads processed training and testing data and labels.

    Args:
        processed_data_path (str): The absolute path to the directory containing the processed data files.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    logging.info(f"Loading processed data from {processed_data_path}")

    try:
        with open(os.path.join(processed_data_path, 'X_train_processed.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(processed_data_path, 'X_test_processed.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        with open(os.path.join(processed_data_path, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(processed_data_path, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        logging.info("Successfully loaded all processed data files.")
    except FileNotFoundError as e:
        logging.error(f"Error loading data file: {e}")
        raise

    # Data Validation Checks
    logging.info("Performing data validation checks...")

    # 1. Shape Consistency
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Shape mismatch: X_train has {X_train.shape[0]} samples, but y_train has {y_train.shape[0]}.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Shape mismatch: X_test has {X_test.shape[0]} samples, but y_test has {y_test.shape[0]}.")
    logging.info("Shape consistency check passed.")

    # 2. Missing Values Check
    if X_train.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in X_train.")
    if X_test.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in X_test.")
    logging.info("Missing values check completed.")

    # 3. Data Type Verification
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test should be pandas DataFrames.")
    if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
        raise TypeError("y_train and y_test should be pandas Series.")
    logging.info("Data type verification passed.")

    logging.info("Data loading and validation complete.")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage:
    # This assumes the script is run from the root of the project directory.
    # You might need to adjust the path depending on your execution context.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'Data', 'processed')
    
    try:
        X_train, X_test, y_train, y_test = load_processed_data(data_path)
        print("Data loaded successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"An error occurred: {e}")
