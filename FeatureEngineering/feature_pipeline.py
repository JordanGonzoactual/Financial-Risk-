import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class FeaturePipeline:
    def __init__(self, 
                 input_filepath,
                 output_dir='Data/processed',
                 artifacts_dir='FeatureEngineering/artifacts',
                 test_size=0.2,
                 random_state=42,
                 target_column=None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            input_filepath (str): Path to the raw CSV data file
            output_dir (str): Directory to save processed data
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
            target_column (str): Name of the target variable column
        """
        self.input_filepath = input_filepath
        self.output_dir = output_dir
        self.artifacts_dir = artifacts_dir
        self.test_size = test_size
        self.random_state = random_state
        self.target_column = target_column
        self.logger = self._setup_logger()
        
        # Ensure output and artifacts directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessing_pipeline = None
        self.feature_names = None
        
    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Prevent duplicate handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

        # Create file handler
        fh = logging.FileHandler('feature_engineering.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
            
        return logger
    
    def load_data(self):
        """
        Load data from CSV file and perform basic data quality checks.
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        self.logger.info(f"Loading data from {self.input_filepath}")
        
        try:
            data = pd.read_csv(self.input_filepath)
            
            # Basic data quality checks
            self.logger.info(f"Data shape: {data.shape}")
            self.logger.info(f"Missing values: {data.isnull().sum().sum()}")
            
            if data.shape[0] == 0:
                raise ValueError("The dataset is empty")
                
            if self.target_column and self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the dataset")
            
            self.data = data
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_preprocessing_pipeline(self, pipeline_builder):
        """
        Create the preprocessing pipeline using the provided pipeline builder.
        
        Args:
            pipeline_builder: An object with a build_pipeline method that returns a sklearn Pipeline
        
        Returns:
            sklearn.pipeline.Pipeline: The preprocessing pipeline
        """
        self.logger.info("Creating preprocessing pipeline")
        try:
            self.preprocessing_pipeline = pipeline_builder.build_pipeline()
            return self.preprocessing_pipeline
        except Exception as e:
            self.logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise
    
    def split_data(self):
        """
        Split the data into training and test sets.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if self.target_column is None:
            raise ValueError("Target column not specified.")
            
        self.logger.info(f"Splitting data with test_size={self.test_size} for regression target {self.target_column}")
        
        try:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            # Save original feature names
            self.feature_names = X.columns.tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Validate data distribution in splits
            train_stats = y_train.describe()
            test_stats = y_test.describe()

            self.logger.info(f"Training set shape: {X_train.shape}")
            self.logger.info(f"Test set shape: {X_test.shape}")
            self.logger.info(f"Training set target statistics:\n{train_stats}")
            self.logger.info(f"Test set target statistics:\n{test_stats}")
            
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def _log_nan_info(self, data, step_name):
        """Helper function to check and log NaNs in a dataset."""
        self.logger.info(f"Checking for NaNs after step: {step_name}")

        # Convert numpy array to DataFrame for consistent checking
        if isinstance(data, np.ndarray):
            # We lose original column names here, but can still check for NaNs
            df = pd.DataFrame(data)
        else:
            df = data

        # Check for NaNs
        nan_total = df.isna().sum().sum()
        if nan_total > 0:
            nan_counts_per_col = df.isna().sum()
            nan_cols = nan_counts_per_col[nan_counts_per_col > 0]
            self.logger.warning(f'Found {nan_total} NaNs in {len(nan_cols)} columns after {step_name}.')
            for col, count in nan_cols.items():
                self.logger.warning(f'  - Column "{col}": {count} NaNs ({count / len(df) * 100:.2f}%)')

        # Check for excessive zeros
        zero_counts_per_col = (df == 0).sum()
        zero_cols = zero_counts_per_col[zero_counts_per_col > 0]
        if not zero_cols.empty:
            self.logger.info(f'Found zeros in {len(zero_cols)} columns after {step_name}.')
            for col, count in zero_cols.items():
                zero_percentage = (count / len(df)) * 100
                if zero_percentage > 10:
                    self.logger.warning(f'  - Column "{col}": {count} zeros ({zero_percentage:.2f}%) - EXCEEDS 10% THRESHOLD')
                else:
                    self.logger.info(f'  - Column "{col}": {count} zeros ({zero_percentage:.2f}%)')
        else:
            self.logger.info("No NaN values found.")

    def process_data(self):
        """
        Apply the preprocessing pipeline to the training and test data, with intermediate NaN checks.
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not created.")
            
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        self.logger.info("Applying preprocessing pipeline with intermediate NaN checks")
        
        try:
            # Fit the entire pipeline on the training data first.
            # This ensures all transformers are fitted correctly and sequentially.
            self.logger.info("Fitting the entire pipeline on training data...")
            self.preprocessing_pipeline.fit(self.X_train, self.y_train)
            self.logger.info("Pipeline fitting complete.")

            # --- Inspect training data transformation step-by-step ---
            self.logger.info("\n--- Inspecting intermediate steps on TRAINING data ---")
            X_intermediate = self.X_train.copy()
            for name, transformer in self.preprocessing_pipeline.steps:
                self.logger.info(f"Transforming training data with step: {name}")
                X_intermediate = transformer.transform(X_intermediate)
                self._log_nan_info(X_intermediate, f"After '{name}' on training data")
            X_train_processed = X_intermediate

            # --- Transform test data using the fully fitted pipeline ---
            self.logger.info("\n--- Inspecting intermediate steps on TEST data ---")
            X_test_intermediate = self.X_test.copy()
            for name, transformer in self.preprocessing_pipeline.steps:
                self.logger.info(f"Transforming test data with step: {name}")
                X_test_intermediate = transformer.transform(X_test_intermediate)
                self._log_nan_info(X_test_intermediate, f"After '{name}' on test data")
            X_test_processed = X_test_intermediate
            self.logger.info("Test data transformation complete.")
            
            self.logger.info(f"Processed training data shape: {X_train_processed.shape}")
            self.logger.info(f"Processed test data shape: {X_test_processed.shape}")
            
            return X_train_processed, X_test_processed
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
    
    def save_transformation_artifacts(self):
        """
        Save the fitted transformation components for the inference pipeline.
        """
        self.logger.info(f"Saving transformation artifacts to {self.artifacts_dir}")

        try:
            # Save the entire pipeline for auditing and debugging
            with open(os.path.join(self.artifacts_dir, 'preprocessing_pipeline.pkl'), 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)

            # Extract and save individual transformers
            numeric_transformer = self.preprocessing_pipeline.named_steps['polynomial_features']
            with open(os.path.join(self.artifacts_dir, 'numeric_scalers.pkl'), 'wb') as f:
                pickle.dump(numeric_transformer, f)

            categorical_transformer = self.preprocessing_pipeline.named_steps['encoding_features']
            with open(os.path.join(self.artifacts_dir, 'categorical_encoders.pkl'), 'wb') as f:
                pickle.dump(categorical_transformer, f)

            # Save the final feature names after transformation
            # This requires transforming the training data to get the final column names
            X_train_transformed = self.preprocessing_pipeline.transform(self.X_train)
            final_feature_names = X_train_transformed.columns.tolist()
            with open(os.path.join(self.artifacts_dir, 'feature_names.pkl'), 'wb') as f:
                pickle.dump(final_feature_names, f)

            # Create and save transformation metadata
            numeric_cols = self.X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            transformation_metadata = {
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols
            }
            with open(os.path.join(self.artifacts_dir, 'transformation_metadata.json'), 'w') as f:
                import json
                json.dump(transformation_metadata, f, indent=4)

            self.logger.info("Successfully saved all transformation artifacts.")

        except Exception as e:
            self.logger.error(f"Error saving transformation artifacts: {str(e)}")
            raise

    def save_processed_data(self, X_train_processed, X_test_processed):
        """
        Save the processed datasets and pipeline to pickle files.
        
        Args:
            X_train_processed: Processed training features
            X_test_processed: Processed test features
        """
        self.logger.info(f"Saving processed data to {self.output_dir}")
        
        try:
            # Save processed data
            with open(os.path.join(self.output_dir, 'X_train_processed.pkl'), 'wb') as f:
                pickle.dump(X_train_processed, f)
                
            with open(os.path.join(self.output_dir, 'X_test_processed.pkl'), 'wb') as f:
                pickle.dump(X_test_processed, f)
                
    
            # Save target variables
            with open(os.path.join(self.output_dir, 'y_train.pkl'), 'wb') as f:
                pickle.dump(self.y_train, f)
                
            with open(os.path.join(self.output_dir, 'y_test.pkl'), 'wb') as f:
                pickle.dump(self.y_test, f)
                
            # Save the fitted pipeline
            with open(os.path.join(self.output_dir, 'preprocessing_pipeline.pkl'), 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)
                
            # Save feature names
            with open(os.path.join(self.output_dir, 'feature_names.pkl'), 'wb') as f:
                pickle.dump(self.feature_names, f)
                
            self.logger.info("Successfully saved all processed data and pipeline")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def generate_summary_report(self, X_train_processed, X_test_processed):
        """
        Generate a summary report of the feature engineering results.
        
        Args:
            X_train_processed: Processed training features
            X_test_processed: Processed test features
        
        Returns:
            dict: Summary statistics
        """
        self.logger.info("Generating feature engineering summary report")
        
        try:
            # Determine target statistics based on dtype
            if self.y_train is not None and pd.api.types.is_numeric_dtype(self.y_train):
                train_target_stats = self.y_train.describe().to_dict()
                test_target_stats = self.y_test.describe().to_dict()
            else:
                train_target_stats = self.y_train.value_counts(normalize=True).to_dict() if self.y_train is not None else None
                test_target_stats = self.y_test.value_counts(normalize=True).to_dict() if self.y_test is not None else None

            # Get removed columns from the pipeline
            removed_columns = []
            if self.preprocessing_pipeline is not None and 'column_dropper' in self.preprocessing_pipeline.named_steps:
                removed_columns = self.preprocessing_pipeline.named_steps['column_dropper'].columns_to_drop

            summary = {
                "pipeline_configuration": {
                    "target_variable": self.target_column,
                    "removed_columns": removed_columns
                },
                "raw_data_shape": self.data.shape if self.data is not None else None,
                "train_test_split": {
                    "train_size": len(self.X_train) if self.X_train is not None else 0,
                    "test_size": len(self.X_test) if self.X_test is not None else 0,
                },
                "processed_data": {
                    "train_shape": X_train_processed.shape if X_train_processed is not None else None,
                    "test_shape": X_test_processed.shape if X_test_processed is not None else None
                },
                "target_statistics": {
                    "train": train_target_stats,
                    "test": test_target_stats
                }
            }
            
            # Add pipeline steps information
            if self.preprocessing_pipeline is not None:
                summary["pipeline_steps"] = [step[0] for step in self.preprocessing_pipeline.steps]
            
            # Log summary
            self.logger.info("Feature Engineering Summary:")
            for key, value in summary.items():
                self.logger.info(f"{key}: {value}")
                
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise
    
    def run(self, pipeline_builder):
        """
        Run the full feature engineering pipeline.
        
        Args:
            pipeline_builder: An object with a build_pipeline method that returns a sklearn Pipeline
            
        Returns:
            dict: Summary of the feature engineering process
        """
        try:
            self.logger.info("Starting feature engineering pipeline")
            
            # Load data
            self.load_data()
            
            # Create preprocessing pipeline
            self.create_preprocessing_pipeline(pipeline_builder)
            
            # Split data
            self.split_data()
            
            # Process data
            X_train_processed, X_test_processed = self.process_data()
            
            # Save processed data
            self.save_processed_data(X_train_processed, X_test_processed)

            # Save transformation artifacts for inference
            self.save_transformation_artifacts()
            
            # Generate summary report
            summary = self.generate_summary_report(X_train_processed, X_test_processed)
            
            self.logger.info("Feature engineering pipeline completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise