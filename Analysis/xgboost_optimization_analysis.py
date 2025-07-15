import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from FeatureEngineering.data_loader import load_processed_data

# --- Configuration ---
MLFLOW_EXPERIMENT_NAME = "XGBoost_Risk_Model"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
MODEL_NAME = "xgboost-risk-model-final"

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def find_and_register_best_model():
    """Finds the best trial from the latest pipeline run and registers the model."""
    logging.info(f"--- Finding and Registering Best Model from Experiment: {MLFLOW_EXPERIMENT_NAME} ---")
    
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        client = MlflowClient()

        # 1. Find the most recent 'full_optimization_pipeline' parent run
        parent_runs = mlflow.search_runs(
            filter_string="tags.`mlflow.runName` = 'full_optimization_pipeline'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if parent_runs.empty:
            logging.error("No 'full_optimization_pipeline' run found. Please run the main pipeline first.")
            return None, None

        parent_run_id = parent_runs.iloc[0].run_id
        logging.info(f"Found latest parent pipeline run with ID: {parent_run_id}")

        # 2. Find the best nested trial run within that parent run
        trial_runs = mlflow.search_runs(
            filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            order_by=["metrics.mean_cv_rmse ASC"],
            max_results=1
        )
        if trial_runs.empty:
            logging.error(f"No trial runs found for parent run {parent_run_id}. The pipeline might have failed.")
            return None, None

        best_trial_run = trial_runs.iloc[0]
        best_run_id = best_trial_run.run_id
        best_rmse = best_trial_run['metrics.mean_cv_rmse']
        logging.info(f"Found best trial run {best_run_id} with Mean CV RMSE: {best_rmse:.6f}")

        # 3. Register the model from the best trial
        model_uri = f"runs:/{best_run_id}/xgboost-model"
        logging.info(f"Registering model from URI: {model_uri}")

        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        logging.info(f"Successfully registered model '{MODEL_NAME}' as Version {registered_model.version}.")

        # 4. Add a description to the registered model version
        description = (
            f"Best model from the full optimization pipeline run {parent_run_id}.\n"
            f"Source trial run ID: {best_run_id}.\n"
            f"Mean CV RMSE: {best_rmse:.6f}."
        )
        client.update_model_version(
            name=MODEL_NAME,
            version=registered_model.version,
            description=description
        )
        logging.info(f"Added description to model version {registered_model.version}.")
        
        return MODEL_NAME, registered_model.version

    except Exception as e:
        logging.error(f"An unexpected error occurred during model registration: {e}", exc_info=True)
        return None, None


def generate_performance_report(model_name, model_version):
    """Loads the registered model and evaluates its performance on the test set."""
    if not model_name or not model_version:
        logging.error("Invalid model name or version provided. Skipping performance report.")
        return

    logging.info(f"--- Generating Performance Report for {model_name} Version {model_version} ---")
    
    try:
        # Load test data
        data_path = os.path.join(project_root, 'Data', 'processed')
        _, X_test, _, y_test = load_processed_data(data_path)
        logging.info(f"Test data loaded. Shape: {X_test.shape}")

        # Load the registered model
        model_uri = f"models:/{model_name}/{model_version}"
        logging.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Evaluate the model
        predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        logging.info(f"Model evaluation complete. Test RMSE: {test_rmse:.6f}")

        # Save the performance metric to a file
        report_path = os.path.join(REPORTS_DIR, 'final_model_performance.txt')
        with open(report_path, 'w') as f:
            f.write(f"Performance Report for Model: {model_name}\n")
            f.write(f"Version: {model_version}\n")
            f.write(f"Test Set RMSE: {test_rmse:.6f}\n")
        logging.info(f"Performance report saved to {report_path}")

    except Exception as e:
        logging.error(f"Failed to generate performance report: {e}", exc_info=True)


if __name__ == '__main__':
    logging.info("Starting analysis of the XGBoost optimization pipeline...")

    # 1. Find the best model from the entire pipeline and register it
    model_name, model_version = find_and_register_best_model()

    # 2. Generate a performance report for the newly registered model
    if model_name and model_version:
        generate_performance_report(model_name, model_version)
    else:
        logging.error("Analysis failed because the best model could not be registered.")

    logging.info("\nAnalysis complete. Report saved to the 'Analysis/reports' directory.")
