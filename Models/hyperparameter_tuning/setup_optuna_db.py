import os
import optuna
import sqlite3

# Define the database path relative to this script's location
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = "xgboost_tuning.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
STORAGE_URL = f"sqlite:///{DB_PATH}"

def setup_optuna_database():
    """
    Initializes the Optuna SQLite database file if it doesn't exist
    and tests the connection by creating and deleting a temporary study.
    Returns the storage URL for the database.
    """
    print(f"--- Setting up Optuna Database at: {DB_PATH} ---")
    try:
        # The study creation itself will create the DB file if it's missing.
        temp_study_name = "_database_connectivity_test_"
        
        print("Attempting to create a temporary study to test connection...")
        study = optuna.create_study(
            storage=STORAGE_URL, 
            study_name=temp_study_name, 
            direction='minimize',
            load_if_exists=True # Use load_if_exists to avoid errors on subsequent runs
        )
        print("Temporary study created successfully.")
        
        # Clean up the temporary study
        optuna.delete_study(study_name=temp_study_name, storage=STORAGE_URL)
        print("Temporary study deleted.")
        
        print("\nDatabase connection verified successfully.")
        print(f"Storage URL: {STORAGE_URL}")
        return STORAGE_URL

    except (sqlite3.OperationalError, RuntimeError) as e:
        print(f"\nError setting up or connecting to the database: {e}")
        print("Please check file permissions and ensure the path is correct.")
        return None

def verify_studies():
    """
    Lists all existing studies in the database and their trial counts.
    """
    print("\n--- Verifying Existing Studies ---")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist. Please run setup first.")
        return

    try:
        summaries = optuna.get_all_study_summaries(storage=STORAGE_URL)
        if not summaries:
            print("No studies found in the database.")
            return
        
        print(f"Found {len(summaries)} studies:")
        for summary in summaries:
            print(f"  - Study: {summary.study_name}, Trials: {summary.n_trials}")
            
    except Exception as e:
        print(f"An error occurred while fetching study summaries: {e}")

if __name__ == '__main__':
    setup_optuna_database()
    verify_studies()
