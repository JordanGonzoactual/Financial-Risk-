import os
import pickle
import logging
import sys
from threading import Lock
import pandas as pd
# Force CPU for XGBoost to avoid GPU/CPU device mismatches in production UI
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
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

            # Ensure model uses CPU predictors to avoid GPU device mismatches
            try:
                if hasattr(self.model, 'set_params'):
                    try:
                        self.model.set_params(device='cpu')
                    except Exception:
                        pass
                    try:
                        self.model.set_params(predictor='cpu_predictor')
                    except Exception:
                        pass
                    try:
                        self.model.set_params(tree_method='hist')
                    except Exception:
                        pass
                try:
                    booster = self.model.get_booster()
                    booster.set_param({'predictor': 'cpu_predictor'})
                except Exception:
                    pass
            except Exception:
                # Best-effort CPU selector; ignore if model doesn't support these params
                pass

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

            # Guard against completely invalid input
            if df_raw is None or df_raw.empty or df_raw.shape[1] == 0:
                raise ValueError("Input DataFrame is empty or has no columns")

            # Validate that at least some expected raw columns are present
            try:
                import json as _json
                artifacts_dir = os.path.join(project_root, 'FeatureEngineering', 'artifacts')
                meta_path = os.path.join(artifacts_dir, 'transformation_metadata.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        _meta = _json.load(f)
                    _raw_cols = set(_meta.get('numeric_columns', []) + _meta.get('categorical_columns', []))
                    overlap = set(df_raw.columns) & _raw_cols
                    if len(overlap) == 0:
                        raise ValueError("No valid input columns provided; none match expected raw schema")
            except ValueError:
                # re-raise value errors we deliberately raised
                raise
            except Exception as e:
                logging.warning(f"Could not validate raw columns against metadata: {e}")
            
            # 1. Transform raw data using the preprocessing pipeline
            # The pipeline handles all transformations and schema validation internally.
            df_processed = self.pipeline.transform(df_raw)
            
            # 2. Make predictions
            logging.info("Generating predictions on engineered features...")
            try:
                predictions = self.model.predict(df_processed)
            except Exception as e:
                msg = str(e)
                if 'cuda' in msg.lower() or 'device' in msg.lower():
                    logging.warning(
                        "GPU/device prediction failed; falling back to CPU predictor.",
                        exc_info=True,
                    )
                    try:
                        if hasattr(self.model, 'set_params'):
                            try:
                                self.model.set_params(device='cpu')
                            except Exception:
                                pass
                            try:
                                self.model.set_params(predictor='cpu_predictor')
                            except Exception:
                                pass
                            try:
                                self.model.set_params(tree_method='hist')
                            except Exception:
                                pass
                        try:
                            booster = self.model.get_booster()
                            booster.set_param({'predictor': 'cpu_predictor'})
                        except Exception:
                            pass
                        predictions = self.model.predict(df_processed)
                    except Exception:
                        logging.error("CPU fallback prediction failed.", exc_info=True)
                        raise
                else:
                    raise
            logging.info(f"Successfully generated {len(predictions)} predictions.")
            
            return pd.Series(predictions, index=df_raw.index)

        except (ValueError, KeyError, Exception) as e:
            logging.error(f"Prediction failed during batch processing: {e}", exc_info=True)
            raise

    def health_check(self) -> dict:
        """Performs a health check on the service's components."""
        model_loaded = self.model is not None
        pipeline_loaded = self.pipeline is not None and self.pipeline.pipeline is not None
        
        return {
            'model_loaded': model_loaded,
            'pipeline_loaded': pipeline_loaded,
            'status': 'healthy' if model_loaded and pipeline_loaded else 'unhealthy'
        }
    
    def get_model_info(self) -> dict:
        """Returns information about the loaded model."""
        if self.model is None:
            return {
                'model_loaded': False,
                'model_type': None,
                'model_version': None
            }
        
        return {
            'model_loaded': True,
            'model_type': type(self.model).__name__,
            'model_version': getattr(self.model, 'version', 'unknown')
        }



# Instantiate the singleton on module import
model_service = ModelService()
