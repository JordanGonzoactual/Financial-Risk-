import os
import sys
import logging
import json
import time
import argparse
import subprocess
import requests
from datetime import datetime
#from dotenv import load_dotenv

import optuna
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from mlflow.models import infer_signature

import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# Add FeatureEngineering to Python path to allow direct imports
sys.path.append(os.path.join(project_root, 'FeatureEngineering'))

from data_loader import load_processed_data
from start_mlflow_server import start_mlflow_server
from Models.hyperparameter_tuning.tune_step1_tree_complexity import run_tree_complexity_tuning
from Models.hyperparameter_tuning.tune_step2_gamma import run_gamma_tuning
from Models.hyperparameter_tuning.tune_step3_sampling import run_sampling_tuning
from Models.hyperparameter_tuning.tune_step4_regularization import run_regularization_tuning
from Models.hyperparameter_tuning.tune_step5_learning_rate import run_learning_rate_tuning

# Load environment variables from .env file
# load_dotenv()

# MLflow Configuration
MLFLOW_CONFIG = {
    "enabled": True,
    "experiment_name": "XGBoost_Risk_Model",
    "tracking_uri": "http://localhost:5000",
    "request_delay": 0.1  # seconds
}

class XGBoostOptimizationPipeline:
    """A pipeline for end-to-end XGBoost hyperparameter optimization."""

    def __init__(self, data_path, trials_per_step=50, terminate_on_error=True, reset=False):
        """
        Initializes the optimization pipeline with configuration parameters.

        Args:
            data_path (str): Path to the processed data.
            trials_per_step (int): Number of optimization trials per step.
            terminate_on_error (bool): Flag to terminate the pipeline on error.
        """
        self.data_path = data_path

        self.trials_per_step = trials_per_step
        self.terminate_on_error = terminate_on_error
        self.reset = reset
        self.parent_run_id = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.base_params = None
        self.start_time = None

    def _ensure_mlflow_server_running(self):
        """Ensures the MLflow server is running by calling the startup script."""
        logging.info("Ensuring MLflow server is running...")
        result = start_mlflow_server()
        logging.info(result)
        # If the server was just started, wait a moment for it to initialize
        if "startup process initiated" in result:
            logging.info("Waiting for MLflow server to initialize...")
            time.sleep(10)




    def _load_and_prepare_data(self):
        """Loads and prepares data for the pipeline, creating a train/validation/test split."""
        try:
            # Load the initial train/test split
            X_train_full, self.X_test, y_train_full, self.y_test = load_processed_data(self.data_path)
            logging.info("Initial data loaded successfully.")

            # Create a train/validation split from the full training data
            # The test set remains untouched
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=None
            )
            logging.info(f"Data split into training, validation, and test sets.")
            logging.info(f"Training set size: {self.X_train.shape[0]} samples")
            logging.info(f"Validation set size: {self.X_val.shape[0]} samples")
            logging.info(f"Test set size: {self.X_test.shape[0]} samples")
        except FileNotFoundError:
            logging.error(f"Processed data not found at {self.data_path}. Please run the feature engineering pipeline first.")
            raise

    def _set_base_params(self):
        """Sets the base parameters for XGBoost."""
        self.base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 100, # Initial value, will be tuned
            'learning_rate': 0.01, # Initial value, will be tuned
            'random_state': 42,
            'device': 'cuda'  # Use GPU for training
        }
        logging.info(f"Base parameters set: {json.dumps(self.base_params, indent=2)}")

    def coordinate_optimization_phases(self, parent_run_id=None):
        """Coordinates the different phases of the hyperparameter optimization."""
        logging.info("Coordinating optimization phases...")
        initial_params = self.base_params.copy()
        
        last_completed_step, current_best_params = self._load_state()

        tuning_steps = [
            ('step1_tree_complexity', run_tree_complexity_tuning),
            ('step2_gamma', run_gamma_tuning),
            ('step3_sampling', run_sampling_tuning),
            ('step4_regularization', run_regularization_tuning),
            ('step5_learning_rate', run_learning_rate_tuning)
        ]
        total_steps = len(tuning_steps)

        start_index = 0
        if last_completed_step:
            completed_steps = [step[0] for step in tuning_steps]
            try:
                start_index = completed_steps.index(last_completed_step) + 1
                if start_index < total_steps:
                    logging.info(f"Resuming optimization from Step {start_index + 1}/{total_steps}: {tuning_steps[start_index][0]}")
                else:
                    logging.info("All optimization steps were already completed. Finalizing results.")
                    start_index = total_steps # Ensure loop doesn't run
            except ValueError:
                logging.warning(f"Could not find last completed step '{last_completed_step}'. Starting from the beginning.")

        for i in range(start_index, total_steps):
            step_name, tuning_function = tuning_steps[i]
            logging.info(f"--- Running Step {i + 1}/{total_steps}: {step_name} ---")

            try:
                # Pass the parent_run_id to the tuning function
                new_params, best_score = tuning_function(
                    self.X_train, self.X_val, self.y_train, self.y_val, 
                    current_best_params,
                    n_trials=self.trials_per_step, 
                    parent_run_id=parent_run_id
                )
                validated_params = self._validate_params(new_params)
                current_best_params.update(validated_params)

                logging.info(f"Completed {step_name} with best RMSE: {best_score:.6f}")
                logging.info(f"Current best params after step: {json.dumps(current_best_params, indent=2)}")
                self._save_state(step_name, current_best_params)
            except Exception as e:
                logging.error(f"An error occurred during {step_name}: {e}", exc_info=True)
                if self.terminate_on_error:
                    logging.error("Pipeline stopped due to error. Set terminate_on_error=False to ignore and continue.")
                    return
                else:
                    logging.warning(f"Skipping {step_name} due to error. Continuing with last known good parameters.")

        logging.info("All optimization steps completed.")
        self._summarize_results(initial_params, current_best_params)

        self._save_final_params(current_best_params)

    def _evaluate_final_model(self):
        """
        Trains the final model on combined train+validation data and evaluates it on the test set.
        """
        logging.info("--- Starting Final Model Evaluation on Test Set ---")
        try:
            # Load the best parameters found during optimization
            _, final_params = self._load_state()
            if not final_params or final_params == self.base_params:
                logging.warning("No optimized parameters found. Skipping final evaluation.")
                return

            logging.info("Training final model on combined training and validation data.")
            
            # Combine train and validation sets
            X_full_train = pd.concat([self.X_train, self.X_val])
            y_full_train = pd.concat([self.y_train, self.y_val])

            # Train the final model
            final_model = xgb.XGBRegressor(enable_categorical=True, **final_params)
            final_model.fit(X_full_train, y_full_train)

            logging.info("Evaluating final model on the hold-out test set.")
            test_preds = final_model.predict(self.X_test)

            # Calculate primary metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, test_preds))
            mae = mean_absolute_error(self.y_test, test_preds)
            r2 = r2_score(self.y_test, test_preds)

            test_metrics = {
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2
            }

            logging.info(f"Final Test Metrics: {json.dumps(test_metrics, indent=2)}")
            mlflow.log_metrics(test_metrics)

            # Save metrics to file for EDA script
            model_metadata = {
                "model_type": "XGBoost Optimized",
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                "test_mape": 0.0,  # Calculate if needed
                "optimization_completed": True,
                "final_params": final_params
            }
            
            # Ensure the trained_models directory exists
            trained_models_dir = os.path.join(project_root, 'Models', 'trained_models')
            os.makedirs(trained_models_dir, exist_ok=True)
            
            # Save metadata for EDA script
            metadata_path = os.path.join(trained_models_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logging.info(f"Model metadata saved to {metadata_path}")

            # Log the final model
            logging.info("Logging final model to MLflow.")
            signature = infer_signature(X_full_train, test_preds)
            mlflow.xgboost.log_model(
                xgb_model=final_model,
                artifact_path="final_model",
                signature=signature,
                registered_model_name="xgboost-final-model"
            )

        except Exception as e:
            logging.error(f"An error occurred during final model evaluation: {e}", exc_info=True)




    def _save_final_params(self, final_params):
        """Saves the final optimized parameters to a JSON file."""
        params_path = os.path.join(os.path.dirname(__file__), 'final_optimized_params.json')
        try:
            with open(params_path, 'w') as f:
                json.dump(final_params, f, indent=4)
            logging.info(f"Saved final optimized parameters to {params_path}")
        except Exception as e:
            logging.error(f"Failed to save final parameters: {e}")

    def _save_state(self, step_name, params):
        """Saves the current state of optimization to a JSON file."""
        state = {
            'last_completed_step': step_name,
            'best_params': params
        }
        state_path = os.path.join(os.path.dirname(__file__), 'optimization_state.json')
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            logging.info(f"Saved optimization state after {step_name} to {state_path}")
        except IOError as e:
            logging.error(f"Could not save state file: {e}")

    def _load_state(self):
        """Loads the optimization state from a JSON file if it exists."""
        state_path = os.path.join(os.path.dirname(__file__), 'optimization_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                logging.info(f"Loaded optimization state from {state_path}")
                # Ensure loaded params are merged with base params in case base_params changed
                loaded_params = self.base_params.copy()
                loaded_params.update(state.get('best_params', {}))
                return state.get('last_completed_step'), loaded_params
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Could not load state file: {e}. Starting fresh.")
                return None, self.base_params.copy()
        return None, self.base_params.copy()

    def _validate_params(self, params):
        """Validates the returned parameters against a list of known XGBoost parameters."""
        known_params = [
            'objective', 'eval_metric', 'tree_method', 'gpu_id', 'random_state',
            'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree',
            'colsample_bylevel', 'reg_alpha', 'reg_lambda', 'learning_rate', 'n_estimators'
        ]
        validated_params = params.copy()
        for key in list(validated_params.keys()): # Use list to allow deletion during iteration
            if key not in known_params:
                logging.warning(f"Unexpected parameter key returned from tuning step: '{key}'. This parameter will be ignored.")
                del validated_params[key]
        return validated_params

    def _summarize_results(self, initial_params, final_params):
        """Logs a summary of the parameter changes."""
        logging.info("--- Optimization Summary ---")
        logging.info("Initial Parameters:")
        logging.info(json.dumps(initial_params, indent=2))
        logging.info("\nFinal Optimized Parameters:")
        logging.info(json.dumps(final_params, indent=2))
        
        improved_params = {k: final_params[k] for k in final_params if k not in initial_params or initial_params[k] != final_params[k]}
        logging.info("\nImproved/Changed Parameters:")
        logging.info(json.dumps(improved_params, indent=2))

    def reset_state(self):
        """Deletes the optimization state file to allow a fresh run."""
        state_path = os.path.join(os.path.dirname(__file__), 'optimization_state.json')
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
                logging.info(f"Successfully removed optimization state file: {state_path}")
            except OSError as e:
                logging.error(f"Error removing state file {state_path}: {e}")
        else:
            logging.info("No optimization state file to reset.")

    def run(self):
        """Main method to run the entire optimization pipeline.""" 
        self._ensure_mlflow_server_running()
        self.start_time = time.time()

        if self.reset:
            self.reset_state()
            logging.info("Reset flag is set. Starting a fresh optimization run.")

        if MLFLOW_CONFIG["enabled"]:
            try:
                mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
                mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
                with mlflow.start_run(run_name="full_optimization_pipeline") as parent_run:
                    self.parent_run_id = parent_run.info.run_id
                    logging.info(f"Started parent MLflow run with ID: {self.parent_run_id}")
                    mlflow.log_param("trials_per_step", self.trials_per_step)

                    self._load_and_prepare_data()
                    self._set_base_params()
                    self.coordinate_optimization_phases(parent_run_id=self.parent_run_id)

                    # Final evaluation on the test set
                    self._evaluate_final_model()

                    # Log final parameters to the parent run
                    _, final_params = self._load_state()
                    mlflow.log_params(final_params)
                    logging.info("Logged final optimized parameters to the parent MLflow run.")
            except Exception as e:
                logging.error(f"MLflow parent run failed: {e}. The pipeline will run without MLflow logging.")
                # Run the pipeline without MLflow if the setup fails
                self._run_without_mlflow()
        else:
            logging.info("MLFLOW_CONFIG is disabled. Running pipeline without MLflow logging.")
            self._run_without_mlflow()

        end_time = time.time()
        duration = (end_time - self.start_time) / 60
        logging.info(f"Total optimization pipeline finished in {duration:.2f} minutes.")

    def _run_without_mlflow(self):
        """Executes the optimization pipeline without any MLflow logging."""
        self._load_and_prepare_data()
        self._set_base_params()
        self.coordinate_optimization_phases()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XGBoost Hyperparameter Optimization Pipeline")
    parser.add_argument('--reset', action='store_true', help='Reset optimization state and start a fresh run.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Instructions for the user
    logging.info("--- XGBoost Hyperparameter Optimization Pipeline ---")
    logging.info("This script will guide you through a multi-step process to tune XGBoost hyperparameters.")
    logging.info("-----------------------------------------------------")

    # Configuration
    DATA_PATH = os.path.join(project_root, 'Data', 'processed')
    TRIALS_PER_STEP = 75 # Adjust as needed

    try:
        pipeline = XGBoostOptimizationPipeline(
            data_path=DATA_PATH, 
            trials_per_step=TRIALS_PER_STEP,
            reset=args.reset
        )
        pipeline.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
