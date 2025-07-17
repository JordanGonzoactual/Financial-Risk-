import os
import sys
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from FeatureEngineering.data_loader import load_processed_data

mlflow.set_tracking_uri("http://localhost:5000")

# --- Configuration ---
MLFLOW_EXPERIMENT_NAME = "XGBoost_Risk_Model"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
MODEL_NAME = "xgboost-final-model"

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_performance_report():
    """Loads the 'challenger' model, evaluates performance, and generates diagnostic plots."""
    model_alias = "champion"
    logging.info(f"--- Generating Performance Report for {MODEL_NAME} with Alias '{model_alias}' ---")
    
    try:
        # Load test data
        data_path = os.path.join(project_root, 'Data', 'processed')
        _, X_test, _, y_test = load_processed_data(data_path)
        logging.info(f"Test data loaded. Shape: {X_test.shape}")

        # Load the registered model using its alias
        model_uri = f"models:/{MODEL_NAME}@{model_alias}"
        logging.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Evaluate the model
        predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        logging.info(f"Model evaluation complete. Test RMSE: {test_rmse:.6f}")

        # Save the performance metric to a file
        report_path = os.path.join(REPORTS_DIR, 'final_model_performance.txt')
        with open(report_path, 'w') as f:
            f.write(f"Performance Report for Model: {MODEL_NAME}\n")
            f.write(f"Alias: {model_alias}\n")
            f.write(f"Test Set RMSE: {test_rmse:.6f}\n")
        logging.info(f"Performance report saved to {report_path}")

        # --- Generate Diagnostic Plots ---
        logging.info("Generating diagnostic plots...")
        residuals = y_test - predictions

        # 1. Q-Q Plot of Residuals
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        qq_plot_path = os.path.join(REPORTS_DIR, 'residuals_qq_plot.png')
        plt.savefig(qq_plot_path)
        plt.close()
        logging.info(f"Saved Q-Q plot to {qq_plot_path}")

        # 2. Residuals vs. Predicted Values
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs. Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        residuals_plot_path = os.path.join(REPORTS_DIR, 'residuals_vs_predicted_plot.png')
        plt.savefig(residuals_plot_path)
        plt.close()
        logging.info(f"Saved residuals plot to {residuals_plot_path}")

        # 3. Predicted vs. Actual Values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Predicted vs. Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        pred_actual_path = os.path.join(REPORTS_DIR, 'predicted_vs_actual_plot.png')
        plt.savefig(pred_actual_path)
        plt.close()
        logging.info(f"Saved predicted vs. actual plot to {pred_actual_path}")

        # 4. Feature Importance Plot
        try:
            # Load the model using the XGBoost flavor for direct access to feature importances
            logging.info("Loading model with mlflow.xgboost to get feature importances.")
            xgb_model = mlflow.xgboost.load_model(model_uri)

            # Create a DataFrame for feature importances
            importances = pd.DataFrame({
                'feature': X_test.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=True)

            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.barh(importances['feature'], importances['importance'])
            plt.title('Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            feature_importance_path = os.path.join(REPORTS_DIR, 'feature_importance_plot.png')
            plt.savefig(feature_importance_path)
            plt.close()
            logging.info(f"Saved feature importance plot to {feature_importance_path}")

        except Exception as e:
            logging.error(f"Could not generate feature importance plot: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Failed to generate performance report: {e}", exc_info=True)


if __name__ == '__main__':
    logging.info("Starting analysis of the XGBoost 'challenger' model...")

    # Generate a performance report for the 'challenger' model
    generate_performance_report()

    logging.info("\nAnalysis complete. Report saved to the 'Analysis/reports' directory.")
