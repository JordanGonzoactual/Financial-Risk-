import os
import pickle
import pandas as pd

class InferencePipeline:
    """
    Loads the preprocessing pipeline and feature names from artifacts
    and applies the transformations to new data for inference.
    """
    def __init__(self, artifacts_dir='FeatureEngineering/artifacts/'):
        """
        Initializes the InferencePipeline by loading transformation artifacts.

        Args:
            artifacts_dir (str): Directory where transformation artifacts are stored.
        """
        self.artifacts_dir = artifacts_dir
        self.preprocessing_pipeline = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Loads the preprocessing pipeline and feature names from the artifacts directory.
        """
        # Load the preprocessing pipeline
        pipeline_path = os.path.join(self.artifacts_dir, 'preprocessing_pipeline.pkl')
        with open(pipeline_path, 'rb') as f:
            self.preprocessing_pipeline = pickle.load(f)

        # Load the feature names
        feature_names_path = os.path.join(self.artifacts_dir, 'feature_names.pkl')
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the loaded preprocessing pipeline to the incoming DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with features ready for the model.
        """
        # Apply the preprocessing pipeline
        processed_data = self.preprocessing_pipeline.transform(df)

        # Create a DataFrame from the processed data with the correct feature names
        processed_df = pd.DataFrame(processed_data, columns=self.feature_names, index=df.index)

        return processed_df
