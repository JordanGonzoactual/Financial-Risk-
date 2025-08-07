import os
import sys
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from FeatureEngineering.data_loader import load_processed_data
from Models.hyperparameter_tuning.mlflow_setup import get_or_create_experiment

def train_and_evaluate_final_model():
    """
    Trains, evaluates, and logs the final model using the best hyperparameters.
    """
    # 1. Load Data
    data_path = os.path.join(project_root, 'Data', 'processed')
    try:
        X_train, X_test, y_train, y_test = load_processed_data(data_path)
        logging.info("Successfully loaded training and testing data.")
    except FileNotFoundError:
        logging.error("Processed data not found. Please run the feature engineering pipeline first.")
        return

    # 2. Load Final Hyperparameters
    params_path = os.path.join(project_root, 'Models', 'hyperparameter_tuning', 'final_optimized_params.json')
    try:
        with open(params_path, 'r') as f:
            final_params = json.load(f)
        logging.info(f"Loaded final optimized hyperparameters: {final_params}")
    except FileNotFoundError:
        logging.error(f"Final parameters file not found at {params_path}. Please run the full tuning pipeline first.")
        return

    # Set up MLflow
    experiment_name = "XGBoost_Hyperparameter_Tuning"
    get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Final_Model_Training_and_Evaluation") as run:
        run_id = run.info.run_id
        logging.info(f"Started final training run: {run_id}")
        mlflow.log_params(final_params)

        # 3. Train the Final Model
        logging.info("Training the final model on the full training dataset...")
        final_model = xgb.XGBRegressor(enable_categorical=True, **final_params)
        final_model.fit(X_train, y_train, verbose=True)
        logging.info("Final model training complete.")

        # 4. Evaluate on Test Set
        logging.info("Evaluating the model on the test set...")
        preds = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        logging.info(f"Test Set Performance:")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  R²: {r2:.4f}")
        logging.info(f"  MAPE: {mape:.4f}%")
        
        # Log all metrics to MLflow
        mlflow.log_metrics({
            "test_rmse": rmse, 
            "test_mse": mse, 
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape
        })

        # Save metrics to model_metadata.json for EDA scripts
        model_metadata = {
            "model_type": "XGBoost Final Model",
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "training_completed": True,
            "final_params": final_params,
            "mlflow_run_id": run_id
        }
        
        # Save metadata file
        metadata_path = os.path.join(project_root, 'Models', 'trained_models', 'model_metadata.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        logging.info(f"Model metadata saved to {metadata_path}")

        # Save the trained model as pickle
        model_save_path = os.path.join(project_root, 'Models', 'trained_models', 'final_model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump(final_model, f)
        logging.info(f"Final model saved as pickle to {model_save_path}")

        # 5. Log Model with Full Artifacts
        logging.info("Logging final model to MLflow with full artifacts...")
        signature = infer_signature(X_train, preds)


        mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="production_model",
            signature=signature,
            registered_model_name="xgboost-risk-model-prod-candidate"
        )
        logging.info("Final model logged and registered as 'xgboost-risk-model-prod-candidate'.")


if __name__ == '__main__':
    train_and_evaluate_final_model()
