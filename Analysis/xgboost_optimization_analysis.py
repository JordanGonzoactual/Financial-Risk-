import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import optuna
import pandas as pd
import matplotlib.pyplot as plt
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
# Construct the absolute path to the database
db_path = os.path.join(project_root, 'Models', 'hyperparameter_tuning', 'xgboost_tuning.db')
OPTUNA_STORAGE_URL = f"sqlite:///{db_path}"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def analyze_optuna_study(study_name):
    """Loads an Optuna study and generates all standard visualization plots."""
    logging.info(f"--- Analyzing Optuna Study: {study_name} ---")
    try:
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE_URL)
    except KeyError:
        logging.error(f"Study '{study_name}' not found in the storage. Please run the tuning step first.")
        return

    # Generate and save plots
    plots = {
        'optimization_history': optuna.visualization.plot_optimization_history,
        'param_importances': optuna.visualization.plot_param_importances,
        'slice': optuna.visualization.plot_slice,
        'contour': optuna.visualization.plot_contour,
    }

    for plot_name, plot_func in plots.items():
        try:
            fig = plot_func(study)
            filepath = os.path.join(REPORTS_DIR, f"{study_name}_{plot_name}.html")
            fig.write_html(filepath)
            logging.info(f"Saved {plot_name} plot to {filepath}")
        except (ValueError, RuntimeError) as e:
            logging.warning(f"Could not generate '{plot_name}' for study '{study_name}'. Reason: {e}")

    # Report on trial statuses
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])

    logging.info("Study statistics: ")
    logging.info(f"  Number of finished trials: {len(study.trials)}")
    logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"  Number of complete trials: {len(complete_trials)}")
    logging.info(f"  Number of failed trials: {len(failed_trials)}")

def register_final_model():
    """Finds the best run from the final tuning step and registers it as the final model."""
    logging.info("--- Registering Final Model ---")
    try:
        # Set the experiment and log its details
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            logging.error(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Aborting model registration.")
            return
        logging.info(f"Searching experiment: '{experiment.name}' (ID: {experiment.experiment_id})")

        client = MlflowClient()

        # --- Debugging Step: List all runs in the experiment ---
        logging.info("--- [Debug] Listing all available runs in the experiment ---")
        all_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        if all_runs.empty:
            logging.warning("[Debug] No runs found in this experiment at all.")
        else:
            logging.info(f"[Debug] Found {len(all_runs)} total runs. Displaying run info:")
            # Print relevant columns for cleaner logging
            relevant_cols = ['run_id', 'start_time', 'tags.mlflow.runName', 'metrics.val_rmse']
            display_runs = all_runs[[col for col in relevant_cols if col in all_runs.columns]]
            logging.info("\n" + display_runs.to_string())
        logging.info("--- [Debug] End of run list ---")

        # Search for the best run from the learning rate tuning step (step 5)
        # The run name is used for filtering as tags were not consistently applied.
        filter_query = "tags.\"mlflow.runName\" LIKE '%step5_learning_rate%'"
        logging.info(f"Searching for the best run using filter: \"{filter_query}\"...")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_query,
            order_by=["metrics.val_rmse ASC"],
            max_results=1
        )

        if runs.empty:
            logging.warning(f"No runs found matching the filter. Skipping model registration.")
            logging.warning("Please check if the hyperparameter tuning script ran successfully and logged to MLflow with the correct tags.")
            return

        best_run_id = runs.iloc[0].run_id
        best_rmse = runs.iloc[0]['metrics.val_rmse']
        logging.info(f"Found best run from step 5 with ID: {best_run_id} (Validation RMSE: {best_rmse:.4f})")

        # Register the model
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "xgboost-risk-model-final"
        try:
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            logging.info(f"Successfully registered model '{model_name}' as version {registered_model.version}.")

            # Add a description to the registered model version
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=f"Best model from learning rate tuning (Step 5). Run ID: {best_run_id}. Validation RMSE: {best_rmse:.4f}."
            )

        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                logging.warning(f"Model '{model_name}' already exists. Skipping registration.")
            else:
                logging.error(f"Failed to register model '{model_name}'. Error: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during model registration: {e}")


def generate_performance_report():
    """Queries MLflow to compare model performance across tuning steps."""
    logging.info("--- Generating Final Model Performance Report ---")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load test data for evaluation
    data_path = os.path.join(project_root, 'Data', 'processed')
    _, X_test, _, y_test = load_processed_data(data_path)

    model_uris = {
        "Step 1": "models:/xgboost-risk-model-step1/latest",
        "Step 2": "models:/xgboost-risk-model-step2/latest",
        "Step 3": "models:/xgboost-risk-model-step3/latest",
        "Step 4": "models:/xgboost-risk-model-step4/latest",
        "Final Registered Model": "models:/xgboost-risk-model-final/latest"
    }

    performance_data = []
    for stage, uri in model_uris.items():
        try:
            model = mlflow.pyfunc.load_model(uri)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            performance_data.append({'Stage': stage, 'Test RMSE': rmse})
            logging.info(f"Evaluated {uri}: RMSE = {rmse:.4f}")
        except mlflow.exceptions.MlflowException:
            logging.warning(f"Could not load model from URI: {uri}. It may not be registered yet.")

    if not performance_data:
        logging.error("No models found to generate a performance report.")
        return

    # Create and save performance DataFrame and plot
    df_perf = pd.DataFrame(performance_data)
    df_perf.to_csv(os.path.join(REPORTS_DIR, 'final_performance_comparison.csv'), index=False)
    logging.info(f"Performance comparison table saved to {REPORTS_DIR}/final_performance_comparison.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_perf['Stage'], df_perf['Test RMSE'], color='c')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Model Performance Improvement Across Tuning Steps')
    ax.set_xticklabels(df_perf['Stage'], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'final_performance_comparison.png'))
    logging.info(f"Performance comparison plot saved to {REPORTS_DIR}/final_performance_comparison.png")




def find_latest_study_names(storage_url):
    """Finds the most recent study for each tuning step from the Optuna storage."""
    try:
        all_studies = optuna.get_all_study_summaries(storage=storage_url)
    except Exception as e:
        # This can happen if the DB is corrupt or has schema issues.
        logging.error(f"Could not retrieve studies from the database. Error: {e}")
        return None

    latest_studies = {}
    study_prefixes = [
        "step1_tree_complexity",
        "step2_gamma",
        "step3_sampling",
        "step4_regularization",
        "step5_learning_rate"
    ]

    for prefix in study_prefixes:
        # Find all studies matching the prefix
        candidate_studies = sorted(
            [s for s in all_studies if s.study_name.startswith(prefix)],
            key=lambda s: s.datetime_start,
            reverse=True
        )
        if candidate_studies:
            latest_studies[prefix] = candidate_studies[0].study_name
            logging.info(f"Found latest study for '{prefix}': {latest_studies[prefix]}")
        else:
            logging.warning(f"No studies found for prefix '{prefix}'.")

    return list(latest_studies.values())

if __name__ == '__main__':
    logging.info("Starting comprehensive analysis of the XGBoost optimization pipeline...")

    # --- Analyze Optuna studies for each step ---
    latest_study_names = find_latest_study_names(OPTUNA_STORAGE_URL)
    if latest_study_names:
        for study_name in latest_study_names:
            analyze_optuna_study(study_name)
    else:
        logging.warning("Could not find any Optuna studies to analyze.")

    # --- Register the best model from the final tuning step ---
    register_final_model()

    # --- Generate and save the final performance report ---
    generate_performance_report()

    logging.info("Comprehensive analysis of the XGBoost optimization pipeline finished.")

    db_filepath = os.path.join(project_root, 'Models', 'hyperparameter_tuning', 'xgboost_tuning.db')
    if not os.path.exists(db_filepath):
        logging.error(f"Optuna database not found at '{db_filepath}'.")
        logging.error("Please run the optimization pipeline first to generate the database.")
        sys.exit(1) # Exit the script if the database is not found

    # Dynamically find the latest study names to analyze
    study_names = find_latest_study_names(OPTUNA_STORAGE_URL)

    if not study_names:
        logging.error("Could not find any studies to analyze. Exiting.")
        sys.exit(1)

    # Analyze each Optuna study
    for name in study_names:
        analyze_optuna_study(name)

    # Register the best model from the final step before generating the report
    register_final_model()

    # Generate the final performance report from MLflow
    generate_performance_report()

    logging.info("\nAnalysis complete. All reports and visualizations have been saved to the 'Analysis/reports' directory.")
