import os
import pickle
import pandas as pd
import logging
from typing import Optional, List
import json

# Configure logger
logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    Loads the complete preprocessing pipeline and expected feature names from training 
    artifacts, applies transformations, and validates the output schema.
    """
    def __init__(self, artifacts_dir: str = 'FeatureEngineering/artifacts/'):
        """
        Initializes the InferencePipeline by loading artifacts.

        Args:
            artifacts_dir (str): Directory where artifacts are stored.
        """
        self.artifacts_dir = artifacts_dir
        self.pipeline: Optional[object] = None
        self.expected_features: Optional[List[str]] = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Loads the preprocessing pipeline and expected feature names.
        """
        pipeline_path = os.path.join(self.artifacts_dir, 'preprocessing_pipeline.pkl')
        features_path = os.path.join(self.artifacts_dir, 'feature_names.pkl')

        if not os.path.exists(pipeline_path) or not os.path.exists(features_path):
            raise FileNotFoundError("Pipeline or feature names artifact not found.")

        try:
            with open(pipeline_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            logger.info("Successfully loaded preprocessing pipeline.")

            with open(features_path, 'rb') as f:
                self.expected_features = pickle.load(f)
            logger.info("Successfully loaded expected feature names.")

        except (pickle.UnpicklingError, EOFError) as e:
            raise IOError(f"Failed to load artifacts: {e}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the preprocessing pipeline and validates the output schema.

        Args:
            df (pd.DataFrame): The input DataFrame with raw data.

        Returns:
            pd.DataFrame: The transformed and validated DataFrame.
        """
        if self.pipeline is None or self.expected_features is None:
            raise RuntimeError("Pipeline artifacts are not loaded.")

        # Prefill missing raw columns to avoid empty DataFrame errors in transformers
        try:
            meta_path = os.path.join(self.artifacts_dir, 'transformation_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                numeric_cols = meta.get('numeric_columns', [])
                categorical_cols = meta.get('categorical_columns', [])
                # Add missing numeric columns with 0.0 and categoricals with empty string
                for col in numeric_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                for col in categorical_cols:
                    if col not in df.columns:
                        df[col] = ''
        except Exception as e:
            logger.warning(f"Failed to prefill missing raw columns: {e}")

        logger.info("Transforming raw data with the preprocessing pipeline...")
        df_processed = self.pipeline.transform(df)
        logger.info("Transformation complete. Validating feature schema.")

        # Schema Validation
        actual_features = df_processed.columns.tolist()
        if len(actual_features) != len(self.expected_features):
            msg = f"Feature count mismatch. Expected {len(self.expected_features)}, got {len(actual_features)}."
            logger.error(msg)
            raise ValueError(msg)

        if set(actual_features) != set(self.expected_features):
            missing = set(self.expected_features) - set(actual_features)
            extra = set(actual_features) - set(self.expected_features)
            msg = f"Feature names mismatch. Missing: {missing}, Extra: {extra}"
            logger.error(msg)
            raise ValueError(msg)

        # Ensure order is correct
        df_processed = df_processed[self.expected_features]
        logger.info("Feature schema validation successful.")

        return df_processed
