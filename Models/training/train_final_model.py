import os
import sys
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
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
    params_path = os.path.join(project_root, 'Models', 'hyperparameter_tuning', 'best_params_final.json')
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
        final_model = xgb.XGBRegressor(**final_params)
        final_model.fit(X_train, y_train, eval_metric='rmse', verbose=True)
        logging.info("Final model training complete.")

        # 4. Evaluate on Test Set
        logging.info("Evaluating the model on the test set...")
        preds = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        logging.info(f"Test Set Performance: RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
        mlflow.log_metrics({"test_rmse": rmse, "test_mse": mse, "test_mae": mae})

        # 5. Log Model with Full Artifacts
        logging.info("Logging final model to MLflow with full artifacts...")
        signature = infer_signature(X_train, preds)
        input_example = X_train.head(5).to_dict(orient='split')
        conda_env = {
            'name': 'mlflow-env',
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.8',
                'pip',
                {
                    'pip': [
                        'mlflow',
                        f'xgboost=={xgb.__version__}',
                        f'scikit-learn=={pd.__version__}',
                        f'pandas=={pd.__version__}'
                    ]
                }
            ]
        }

        mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="production_model",
            signature=signature,
            input_example=input_example,
            conda_env=conda_env,
            registered_model_name="xgboost-risk-model-prod-candidate"
        )
        logging.info("Final model logged and registered as 'xgboost-risk-model-prod-candidate'.")

        # 6. Create Model Comparison Artifact
        logging.info("Generating performance comparison artifact...")
        model_versions = {
            "Step 1: Tree Complexity": "xgboost-risk-model-step1",
            "Step 2: Gamma": "xgboost-risk-model-step2",
            "Step 3: Sampling": "xgboost-risk-model-step3",
            "Step 4: Regularization": "xgboost-risk-model-step4",
            "Step 5: Final Model": "xgboost-risk-model-prod-candidate"
        }

        performance_data = {}
        for stage, model_name in model_versions.items():
            try:
                # Load the latest version of each registered model
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.pyfunc.load_model(model_uri)
                
                # Evaluate performance
                stage_preds = model.predict(X_test)
                stage_rmse = np.sqrt(mean_squared_error(y_test, stage_preds))
                performance_data[stage] = stage_rmse
                logging.info(f"Evaluated {model_name} (latest): RMSE = {stage_rmse:.4f}")
            except mlflow.exceptions.MlflowException as e:
                logging.warning(f"Could not load model {model_name}. It might not be registered yet. Error: {e}")

        if performance_data:
            # Create and log the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            stages = list(performance_data.keys())
            rmses = list(performance_data.values())
            ax.bar(stages, rmses, color='skyblue')
            ax.set_ylabel('Test RMSE')
            ax.set_title('Model Performance Improvement Across Tuning Steps')
            ax.set_xticklabels(stages, rotation=45, ha="right")
            plt.tight_layout()
            mlflow.log_figure(fig, "performance_comparison.png")
            logging.info("Performance comparison plot logged to MLflow.")

if __name__ == '__main__':
    train_and_evaluate_final_model()
