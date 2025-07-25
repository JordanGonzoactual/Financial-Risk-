import pandas as pd
import streamlit as st
import io
from .state import AppState
from utils.logging import get_logger
from utils.data_validator import CSVValidator
from .api_client import RiskAssessmentClient

logger = get_logger(__name__)

class AppController:
    def __init__(self):
        self.api_client = RiskAssessmentClient()
    def handle_file_upload(self, uploaded_file):
        # Reset state if a new file is uploaded
        if uploaded_file.name != AppState.get_state('uploaded_file_name'):
            AppState.reset_all() # Simplified state reset
            AppState.set_state('uploaded_file_name', uploaded_file.name)

        try:
            file_content = uploaded_file.getvalue()
            df_preview = None
            csv_content = None

            if uploaded_file.name.endswith('.csv'):
                csv_content = file_content.decode('utf-8')
                # Use a fresh IO stream for reading into pandas
                df_preview = pd.read_csv(io.StringIO(csv_content))
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df_preview = pd.read_excel(io.BytesIO(file_content))
                # Convert to CSV string for API compatibility and storage
                csv_content = df_preview.to_csv(index=False)
            
            AppState.set_state('raw_data', df_preview)
            AppState.set_state('csv_content', csv_content) # Store raw CSV content
            logger.info(f"Successfully loaded and processed {uploaded_file.name}")

        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            st.error(f"Failed to read file: {e}")

    def handle_data_validation(self):
        df = AppState.get_state('raw_data')
        if df is None:
            st.warning("Please upload a file first.")
            return

        validator = CSVValidator(df)
        report = validator.validate()
        AppState.set_state('validation_report', report)

        if report['is_valid']:
            clean_df = validator.get_clean_data()
            AppState.set_state('validated_data', clean_df)
            logger.info("Data validation successful.")
            st.success("Data is valid and has been cleaned!")
        else:
            AppState.set_state('validated_data', None) # Ensure no stale data
            logger.warning("Data validation failed.")
            error_messages = "\n".join(report['errors'])
            st.error(f"Data validation failed with the following errors:\n{error_messages}")

    def handle_assessment_processing(self):
        # We use 'validated_data' to confirm validation passed, but send 'csv_content'
        if AppState.get_state('validated_data') is None:
            st.warning("Please upload and validate your data first.")
            return

        csv_data = AppState.get_state('csv_content')
        if csv_data is None:
            st.error("Could not find the CSV data to process. Please re-upload the file.")
            logger.error("Assessment aborted: 'csv_content' is missing from state.")
            return

        with st.spinner('Running risk assessment... This may take a moment.'):
            results = self.api_client.predict_batch(csv_data)
            if results is not None:
                AppState.set_state('assessment_results', results)
                logger.info("Risk assessment processing finished successfully.")
                st.success("Risk assessment complete!")
            else:
                # Error is already handled and displayed by the client
                logger.error("Risk assessment processing failed.")
