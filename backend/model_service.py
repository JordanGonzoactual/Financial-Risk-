import os
import pickle
import logging
import sys
from threading import Lock
import pandas as pd
import xgboost as xgb
from FeatureEngineering.inference_pipeline import InferencePipeline

# Add project root to Python path for consistent module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

class ModelService:
    """A singleton service to manage the XGBoost model and preprocessing pipeline."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the service by loading artifacts if not already done."""
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    self._load_artifacts()
                    self.initialized = True

    def _load_artifacts(self):
        """Loads the XGBoost model and the complete preprocessing pipeline."""
        logging.info("Initializing ModelService and loading artifacts...")
        self.model = None
        self.pipeline = None

        try:
            # Load XGBoost Model
            trained_models_dir = os.path.join(project_root, 'Models', 'trained_models')
            model_file = os.path.join(trained_models_dir, 'final_model.pkl')
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            logging.info("Successfully loaded XGBoost model.")

            # Load Preprocessing Pipeline
            artifacts_dir = os.path.join(project_root, 'FeatureEngineering', 'artifacts')
            self.pipeline = InferencePipeline(artifacts_dir=artifacts_dir)
            logging.info("Successfully loaded and initialized preprocessing pipeline.")

        except (FileNotFoundError, pickle.UnpicklingError, IOError, Exception) as e:
            logging.critical(f"Failed to load artifacts: {e}", exc_info=True)
            self.model = None
            self.pipeline = None

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.Series:
        """Applies the preprocessing pipeline to raw data and returns predictions."""
        if not self.health_check()['pipeline_loaded']:
            raise RuntimeError("Preprocessing pipeline not loaded, cannot make predictions.")
        if not self.health_check()['model_loaded']:
            raise RuntimeError("Model not loaded, cannot make predictions.")

        try:
            logging.info(f"Processing a batch of {len(df_raw)} records.")
            
            # 1. Transform raw data using the preprocessing pipeline
            # The pipeline handles all transformations and schema validation internally.
            df_processed = self.pipeline.transform(df_raw)
            
            # 2. Make predictions
            logging.info("Generating predictions on engineered features...")
            predictions = self.model.predict(df_processed)
            logging.info(f"Successfully generated {len(predictions)} predictions.")
            
            return pd.Series(predictions, index=df_raw.index)

        except (ValueError, KeyError, Exception) as e:
            logging.error(f"Prediction failed during batch processing: {e}", exc_info=True)
            raise

    def health_check(self) -> dict:
        """Performs a health check on the service's components."""
        return {
            'model_loaded': self.model is not None,
            'pipeline_loaded': self.pipeline is not None and self.pipeline.pipeline is not None
        }



# Instantiate the singleton on module import
model_service = ModelService()
