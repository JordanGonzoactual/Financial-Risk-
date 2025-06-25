"""
MLflow Configuration for Financial Risk Project
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import os
from pathlib import Path

class MLflowConfig:
    def __init__(self, experiment_name="financial_risk_prediction"):
        self.experiment_name = experiment_name
        self.tracking_uri = self._setup_tracking_uri()
        self.setup_mlflow()
    
    def _setup_tracking_uri(self):
        """Setup MLflow tracking URI"""
        # Create MLflow directory if it doesn't exist
        mlflow_dir = Path("MLflow/mlruns")
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Use file-based tracking for simplicity
        return f"file://{mlflow_dir.absolute()}"
    
    def setup_mlflow(self):
        """Initialize MLflow experiment"""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow experiment '{self.experiment_name}' is ready")
        print(f"Tracking URI: {self.tracking_uri}")
        
        return experiment_id
    
    def start_run(self, run_name=None, tags=None):
        """Start an MLflow run"""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_model_metrics(self, model, X_test, y_test, model_name):
        """Log model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        
        if y_pred_proba is not None:
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
        
        # Log model
        if 'xgboost' in str(type(model)).lower():
            mlflow.xgboost.log_model(model, f"{model_name}_model")
        elif 'lightgbm' in str(type(model)).lower():
            mlflow.lightgbm.log_model(model, f"{model_name}_model")
        else:
            mlflow.sklearn.log_model(model, f"{model_name}_model")
    
    def log_hyperparameters(self, params):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

# Usage example
if __name__ == "__main__":
    config = MLflowConfig()
    print("MLflow configuration completed successfully!")
