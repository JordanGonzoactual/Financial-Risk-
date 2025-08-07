"""
MLflow setup utilities for experiment management.
This module provides functions to create and manage MLflow experiments.
"""

import mlflow
import logging
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_or_create_experiment(experiment_name):
    """
    Get an existing MLflow experiment or create a new one if it doesn't exist.
    
    Args:
        experiment_name (str): Name of the experiment to get or create
        
    Returns:
        str: The experiment ID
        
    Raises:
        MlflowException: If there's an error creating or accessing the experiment
    """
    try:
        # Try to get the experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is not None:
            logging.info(f"Found existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
        else:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
            
    except MlflowException as e:
        logging.error(f"MLflow error when getting/creating experiment '{experiment_name}': {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error when getting/creating experiment '{experiment_name}': {e}")
        raise MlflowException(f"Failed to get or create experiment: {e}")

def setup_mlflow_tracking(tracking_uri="http://localhost:5000"):
    """
    Set up MLflow tracking URI.
    
    Args:
        tracking_uri (str): The MLflow tracking server URI
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"MLflow tracking URI set to: {tracking_uri}")
    except Exception as e:
        logging.error(f"Failed to set MLflow tracking URI: {e}")
        raise

def get_experiment_info(experiment_name):
    """
    Get information about an MLflow experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        dict: Experiment information including ID, name, and lifecycle stage
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "lifecycle_stage": experiment.lifecycle_stage,
                "artifact_location": experiment.artifact_location
            }
        else:
            return None
    except Exception as e:
        logging.error(f"Error getting experiment info for '{experiment_name}': {e}")
        return None
