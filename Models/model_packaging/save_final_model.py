import os
import sys
import json
import logging
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pickle
from datetime import datetime
import time

# --- Setup Paths and Logging ---

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import custom modules
from FeatureEngineering.data_loader import load_processed_data
from start_mlflow_server import start_mlflow_server

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

MLFLOW_CONFIG = {
    "experiment_name": "xgboost_hyperparameter_tuning_v2",
    "tracking_uri": "http://localhost:5000",
    "registered_model_name": "financial_risk_model_production"
}

# Define base paths
DATA_PATH = os.path.join(project_root, 'Data', 'processed')
PARAMS_PATH = os.path.join(project_root, 'Models', 'hyperparameter_tuning', 'final_optimized_params.json')
TRAINED_MODELS_DIR = os.path.join(project_root, 'Models', 'trained_models')

# Ensure base artifact directory exists
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def load_optimized_parameters(params_path):
    """Loads optimized hyperparameters from a JSON file."""
    logging.info(f"Loading optimized parameters from: {params_path}")
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        logging.info("Optimized parameters loaded successfully.")
        logging.info(f"Parameters: {json.dumps(params, indent=2)}")
        return params
    except FileNotFoundError:
        logging.error(f"Fatal: Parameter file not found at {params_path}.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Fatal: Could not decode JSON from {params_path}.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Handles division by zero by excluding zero-valued true observations.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero; filter out true values that are zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan  # Return NaN if all true values are zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_performance_metrics(model, X, y, prefix):
    """Calculates and logs performance metrics for a given dataset."""
    logging.info(f"Calculating performance metrics for {prefix} set...")
    predictions = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)

    metrics = {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
        f"{prefix}_mape": mape
    }
    
    logging.info(f"{prefix.capitalize()} Set Metrics: {json.dumps(metrics, indent=2)}")
    return metrics

def create_model_package(model, metrics, run_id, X_test, y_test):
    """Saves the five essential model artifacts to the trained_models directory."""
    logging.info(f"Creating essential model artifacts in: {TRAINED_MODELS_DIR}")
    
    try:
        # 1. Save final_model.pkl
        model_path = os.path.join(TRAINED_MODELS_DIR, 'final_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Saved pickled model to {model_path}")

        # 2. Save model_metadata.json
        metadata_path = os.path.join(TRAINED_MODELS_DIR, 'model_metadata.json')
        metadata = {
            'model_version': datetime.now().strftime("%Y%m%d%H%M%S"),
            'training_date': datetime.now().isoformat(),
            'model_type': 'XGBoost Regressor',
            'feature_count': len(X_test.columns),
            'mlflow_run_id': run_id
        }
        # Add all calculated performance metrics to the metadata
        metadata.update(metrics)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved model metadata to {metadata_path}")

        # 3. Save input_schema.json
        schema_path = os.path.join(TRAINED_MODELS_DIR, 'input_schema.json')
        schema = {
            'required_features': list(X_test.columns),
            'feature_types': {col: str(dtype) for col, dtype in X_test.dtypes.items()}
        }
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        logging.info(f"Saved input schema to {schema_path}")

        # 4. Save feature_importance.csv
        importance_path = os.path.join(TRAINED_MODELS_DIR, 'feature_importance.csv')
        importance_df = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(importance_path, index=False)
        logging.info(f"Saved feature importance to {importance_path}")

        # 5. Save sample_predictions.csv
        predictions_path = os.path.join(TRAINED_MODELS_DIR, 'sample_predictions.csv')
        predictions_df = X_test.copy()
        predictions_df['actual_risk_score'] = y_test
        predictions_df['predicted_risk_score'] = model.predict(X_test)
        predictions_df.head(5).to_csv(predictions_path, index=False)
        logging.info(f"Saved sample predictions to {predictions_path}")

    except IOError as e:
        logging.error(f"Error saving model artifact: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during artifact creation: {e}")
        raise

    return TRAINED_MODELS_DIR


def check_mlflow_connectivity(tracking_uri):
    """Checks if the MLflow tracking server is accessible.

    Args:
        tracking_uri (str): The MLflow tracking URI.

    Returns:
        bool: True if the server is accessible, False otherwise.
    """
    logging.info(f"Checking MLflow server connectivity at {tracking_uri}...")
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        # A lightweight, fast operation to check for server response.
        client.search_experiments(max_results=1)
        logging.info("MLflow server connection successful.")
        return True
    except Exception as e:
        logging.error(f"MLflow server connection failed: {e}")
        logging.warning("Proceeding with local artifact generation only.")
        return False

def train_and_package_model():
    """Main function to train, evaluate, and package the final model."""
    logging.info("--- Starting Final Model Training and Packaging --- ")

    # Start MLflow server and wait for it to initialize
    start_mlflow_server()
    logging.info("Waiting 10 seconds for MLflow server to initialize...")
    time.sleep(10)

    try:
        # Load components
        optimized_params = load_optimized_parameters(PARAMS_PATH)
        X_train, X_test, y_train, y_test = load_processed_data(DATA_PATH)

        # Train the final model
        logging.info("Training final model on the full training set...")
        model = xgb.XGBRegressor(**optimized_params)
        model.fit(X_train, y_train)
        test_metrics = calculate_performance_metrics(model, X_test, y_test, 'test')
   
        logging.info("Initial model training completed.")

        # Incrementally train on the test set
        logging.info("Incrementally training model on the test dataset...")
        model.fit(X_test, y_test, xgb_model=model.get_booster())
        logging.info("Incremental training completed.")
        all_metrics = {**test_metrics}

        # Create local artifacts first, ensuring they are always available
        package_path = create_model_package(model, all_metrics, "local-only", X_test, y_test)
        logging.info(f"Successfully created local model artifacts at: {package_path}")

        # Proceed with MLflow logging only if the server is available
        if check_mlflow_connectivity(MLFLOW_CONFIG["tracking_uri"]):
            try:
                mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
                mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

                with mlflow.start_run(run_name="final_model_packaging_v2") as run:
                    run_id = run.info.run_id
                    logging.info(f"MLflow Run Started for persistence: {run_id}")

                    # Log parameters and metrics
                    mlflow.log_params(optimized_params)
                    mlflow.log_metrics(all_metrics)
                    logging.info(f"Successfully logged params and metrics to run ID: {run_id}")

                    # Log artifacts
                    mlflow.log_artifacts(package_path, artifact_path="model_package")
                    logging.info(f"Successfully logged all artifacts from {package_path} to run ID: {run_id}")

                    # Register model and add metadata
                    X_full = pd.concat([X_train, X_test])
                    signature = infer_signature(X_full, model.predict(X_full))
                    model_info = mlflow.xgboost.log_model(
                        xgb_model=model,
                        artifact_path="xgboost-model",
                        signature=signature,
                        registered_model_name=MLFLOW_CONFIG["registered_model_name"]
                    )
                    version = model_info.registered_model_version
                    logging.info(f"Successfully registered model '{MLFLOW_CONFIG['registered_model_name']}' as version {version}")

                    # Add description and tags
                    client = MlflowClient()
                    client.update_model_version(
                        name=MLFLOW_CONFIG["registered_model_name"],
                        version=version,
                        description="Production candidate model trained on full dataset."
                    )
                    client.set_model_version_tag(name=MLFLOW_CONFIG["registered_model_name"], version=version, key="validation_status", value="pending")
                    logging.info(f"Successfully added description and tags to model version {version}.")

            except mlflow.exceptions.MlflowException as e:
                logging.error(f"An MLflow error occurred during database persistence: {e}", exc_info=True)
                logging.warning("Local artifacts were preserved.")

    except FileNotFoundError as e:
        logging.error(f"A required file was not found: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected fatal error occurred: {e}", exc_info=True)
        sys.exit(1)
    except FileNotFoundError as e:
        logging.error(f"A required file was not found: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during model packaging: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Model Packaging Script Finished Successfully ---")

if __name__ == '__main__':
    try:
        train_and_package_model()
    except Exception as e:
        logging.error(f"Script failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)
