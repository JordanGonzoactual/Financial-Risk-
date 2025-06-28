import os
import json
import logging
import time
import pandas as pd

from feature_pipeline import FeaturePipeline
from pipeline_builder import build_pipeline_from_dataframe

# --- Configuration ---
INPUT_FILE = 'Data/raw/Loan.csv'
OUTPUT_DIR = 'Data/processed'
TARGET_COLUMN = 'RiskScore'
COLUMNS_TO_REMOVE = ['LoanApproved']
LOG_FILE = 'Logs/data_processor.log'
SUMMARY_REPORT_TXT = 'Results/feature_engineering_summary.txt'
SUMMARY_REPORT_JSON = 'Results/feature_engineering_summary.json'

# --- Logger Setup ---
def setup_logger():
    """Set up a logger that outputs to both console and a log file."""
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(LOG_FILE, mode='w')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

# --- Pipeline Builder Wrapper ---
class PipelineBuilder:
    """A wrapper to make pipeline_builder.py compatible with FeaturePipeline."""
    def __init__(self, df, columns_to_drop=None, **kwargs):
        self.df = df
        self.columns_to_drop = columns_to_drop
        self.kwargs = kwargs

    def build_pipeline(self):
        """Builds the pipeline using the dataframe."""
        return build_pipeline_from_dataframe(
            self.df, 
            columns_to_drop=self.columns_to_drop, 
            **self.kwargs
        )

# --- Main Execution ---
def main():
    """Main function to run the feature engineering workflow."""
    logger = setup_logger()
    start_time = time.time()

    try:
        # --- Validation ---
        logger.info("--- Starting Validation ---")
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Input file not found: {INPUT_FILE}")
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not os.access(OUTPUT_DIR, os.W_OK):
            logger.error(f"Output directory is not writable: {OUTPUT_DIR}")
            raise PermissionError(f"Output directory is not writable: {OUTPUT_DIR}")
        logger.info("Validation successful.")

        # --- Feature Engineering ---
        logger.info("--- Initializing Feature Engineering Pipeline ---")
        feature_pipeline = FeaturePipeline(
            input_filepath=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            target_column=TARGET_COLUMN
        )

        # Load data to build the pipeline dynamically
        logger.info("Loading data to define pipeline structure...")
        temp_data = pd.read_csv(INPUT_FILE)

        # --- Data Validation ---
        logger.info("--- Validating Data ---")
        
        # 1. Validate Target Column
        if TARGET_COLUMN not in temp_data.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        
        logger.info(f"Target column '{TARGET_COLUMN}' found.")
        if pd.api.types.is_numeric_dtype(temp_data[TARGET_COLUMN]):
            logger.info(f"Target column distribution:\n{temp_data[TARGET_COLUMN].describe()}")
        else:
            logger.info(f"Target column distribution:\n{temp_data[TARGET_COLUMN].value_counts(normalize=True)}")

        # 2. Validate Columns to Remove
        if COLUMNS_TO_REMOVE:
            actual_columns_to_remove = [col for col in COLUMNS_TO_REMOVE if col in temp_data.columns]
            missing_columns = set(COLUMNS_TO_REMOVE) - set(actual_columns_to_remove)
            
            if actual_columns_to_remove:
                logger.info(f"Columns to be removed by the pipeline: {actual_columns_to_remove}")
            
            if missing_columns:
                logger.warning(f"Columns specified for removal but not found in dataset: {list(missing_columns)}")

        initial_memory_usage = temp_data.memory_usage(deep=True).sum()

        # Create the pipeline builder, passing columns to be removed by the pipeline
        pipeline_builder = PipelineBuilder(
            temp_data.drop(columns=[TARGET_COLUMN], errors='ignore'),
            columns_to_drop=COLUMNS_TO_REMOVE
        )

        logger.info("--- Running Feature Engineering Workflow ---")
        summary = feature_pipeline.run(pipeline_builder)

        # --- Augment Summary Report ---
        logger.info("--- Generating Final Summary Report ---")
        end_time = time.time()
        final_memory_usage = pd.read_pickle(os.path.join(OUTPUT_DIR, 'X_train_processed.pkl')).memory_usage(deep=True).sum()

        summary['performance'] = {
            'total_execution_time_seconds': round(end_time - start_time, 2)
        }
        summary['memory_optimization'] = {
            'initial_memory_usage_mb': round(initial_memory_usage / (1024 * 1024), 2),
            'final_memory_usage_mb': round(final_memory_usage / (1024 * 1024), 2),
            'reduction_percentage': round((1 - final_memory_usage / initial_memory_usage) * 100, 2) if initial_memory_usage > 0 else 0
        }

        # --- Save Summary Report ---
        os.makedirs(os.path.dirname(SUMMARY_REPORT_TXT), exist_ok=True)

        # Save as JSON
        with open(SUMMARY_REPORT_JSON, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Summary report saved to {SUMMARY_REPORT_JSON}")

        # Save as TXT
        with open(SUMMARY_REPORT_TXT, 'w') as f:
            for key, value in summary.items():
                f.write(f"--- {key.replace('_', ' ').upper()} ---\n")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        f.write(f"{sub_key}: {sub_value}\n")
                else:
                    f.write(f"{value}\n")
                f.write('\n')
        logger.info(f"Summary report saved to {SUMMARY_REPORT_TXT}")

        logger.info("--- Feature Engineering Workflow Completed Successfully ---")

    except FileNotFoundError as e:
        logger.error(f"Setup Error: {e}")
    except PermissionError as e:
        logger.error(f"Permissions Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()