import os
import sys
import json
import logging
import argparse
import time
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

# Add project root to path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from FeatureEngineering.data_loader import load_processed_data
from start_mlflow_server import start_mlflow_server

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MLflow Configuration - should match the optimization script
MLFLOW_CONFIG = {
    "enabled": True,
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment_name": "xgboost_hyperparameter_tuning_v2"
}

class ModelRegistrar:
    """Handles the training, validation, and registration of the final model."""

    def __init__(self, params_path, data_path, model_name):
        self.params_path = params_path
        self.data_path = data_path
        self.model_name = model_name
        self.final_params = None
        self.model = None
        self.X_train, self.y_train = None, None

    def _load_final_params(self):
        """Loads the final optimized hyperparameters from the JSON file."""
        logging.info(f"Loading final parameters from {self.params_path}")
        try:
            with open(self.params_path, 'r') as f:
                self.final_params = json.load(f)
            logging.info("Successfully loaded final parameters.")
        except FileNotFoundError:
            logging.error(f"Fatal: Parameter file not found at {self.params_path}. Exiting.")
            raise
        except json.JSONDecodeError:
            logging.error(f"Fatal: Could not decode JSON from {self.params_path}. Exiting.")
            raise

    def _load_and_prepare_data(self):
        """Loads and prepares the full training data."""
        logging.info(f"Loading processed data from {self.data_path}")
        try:
            X_train, X_test, y_train, y_test = load_processed_data(self.data_path)
            # For the final model, we train on the entire available training dataset.
            # Combine train and test sets to get the full dataset
            self.X_train = pd.concat([X_train, X_test], ignore_index=True)
            self.y_train = pd.concat([y_train, y_test], ignore_index=True)
            logging.info(f"Training data loaded successfully. Shape: {self.X_train.shape}")
        except FileNotFoundError:
            logging.error(f"Fatal: Processed data files not found in {self.data_path}. Exiting.")
            raise

    def _train_production_model(self):
        """Trains the final XGBoost model on the full training data."""
        if not self.final_params or self.X_train is None:
            logging.error("Fatal: Parameters or data not loaded. Cannot train model.")
            return

        logging.info("Training the final production model...")
        self.model = xgb.XGBRegressor(enable_categorical=True, **self.final_params)
        self.model.fit(self.X_train, self.y_train, verbose=False)
        logging.info("Final model training completed.")

    def _validate_model(self):
        """Performs a simple validation check on the trained model."""
        if self.model is None:
            logging.error("Model not trained. Validation failed.")
            return False
        
        # Simple sanity check: ensure predictions are not all null or constant
        preds = self.model.predict(self.X_train.head(10))
        if pd.Series(preds).isnull().all():
            logging.error("Validation failed: Model returns all null predictions.")
            return False
        if pd.Series(preds).nunique() == 1:
            logging.warning("Validation check: Model returns constant predictions on sample data.")
        
        logging.info("Model passed basic validation checks.")
        return True

    def _register_model(self):
        """Logs and registers the model to MLflow Model Registry."""
        if not self._validate_model():
            logging.error("Model did not pass validation. Skipping registration.")
            return

        # Start MLflow server and wait for it to initialize
        logging.info("Starting MLflow server...")
        start_mlflow_server()
        logging.info("Waiting 15 seconds for MLflow server to initialize...")
        time.sleep(15)

        logging.info(f"Registering model '{self.model_name}' to MLflow...")
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

        with mlflow.start_run(run_name="final_model_registration") as run:
            mlflow.log_params(self.final_params)
            
            # Infer signature
            signature = infer_signature(self.X_train, self.model.predict(self.X_train))
            
            # Log and register the model
            model_info = mlflow.xgboost.log_model(
                xgb_model=self.model,
                artifact_path="production_model",
                signature=signature,
                registered_model_name=self.model_name
            )
            
            logging.info(f"Model registered successfully. Version: {model_info.registered_model_version}")

            # Add tags and description to the registered model version
            client = MlflowClient()
            client.update_model_version(
                name=self.model_name,
                version=model_info.registered_model_version,
                description="This is the final production-candidate model trained on the full dataset."
            )
            client.set_model_version_tag(name=self.model_name, version=model_info.registered_model_version, key="status", value="production-candidate")
            client.set_model_version_tag(name=self.model_name, version=model_info.registered_model_version, key="validated", value="true")
            
            logging.info("Added tags and description to the new model version.")

    def run(self):
        """Executes the full model registration pipeline."""
        try:
            self._load_final_params()
            self._load_and_prepare_data()
            self._train_production_model()
            self._register_model()
            logging.info("Model registration process completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during the model registration process: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Register Final XGBoost Model.")
    parser.add_argument('--params-path', type=str, default=os.path.join(os.path.dirname(__file__), 'final_optimized_params.json'), help='Path to the final hyperparameters JSON file.')
    parser.add_argument('--data-path', type=str, default=os.path.join(project_root, 'Data', 'processed'), help='Path to the directory containing training data.')
    parser.add_argument('--model-name', type=str, default='xgboost-final-model', help='Name for the registered model in MLflow.')

    args = parser.parse_args()

    registrar = ModelRegistrar(
        params_path=args.params_path,
        data_path=args.data_path,
        model_name=args.model_name
    )
    registrar.run()
