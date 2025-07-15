import os
import optuna
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the database path relative to this script's location
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = "xgboost_tuning.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
STORAGE_URL = f"sqlite:///{DB_PATH}"

# Define the sequence of studies to check
STUDY_NAMES = [
    "step1_tree_complexity",
    "step2_gamma",
    "step3_sampling",
    "step4_regularization",
    "step5_learning_rate"
]

def check_optimization_progress():
    """
    Iterates through all study names and displays the number of completed trials
    and the best RMSE for each existing study.
    """
    print("--- Checking Optuna Optimization Progress ---")
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at {DB_PATH}. Please run the setup script or a tuning step first.")
        return

    for study_name in STUDY_NAMES:
        try:
            study = optuna.load_study(study_name=study_name, storage=STORAGE_URL)
            
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not completed_trials:
                print(f"\nStudy '{study_name}': Found, but has 0 completed trials.")
                continue

            best_trial = study.best_trial
            print(f"\nStudy '{study_name}':")
            print(f"  - Completed Trials: {len(completed_trials)}")
            print(f"  - Best RMSE: {best_trial.value:.6f}")
            print(f"  - Best Parameters: {best_trial.params}")

        except KeyError:
            # This error is raised if the study does not exist in the database yet
            print(f"\nStudy '{study_name}': Not found. Has not been run yet.")
        except Exception as e:
            print(f"\nAn error occurred while checking study '{study_name}': {e}")

if __name__ == '__main__':
    check_optimization_progress()
