import pandas as pd
import requests
import streamlit as st
import io
from utils.logging import get_logger

logger = get_logger(__name__)

class RiskAssessmentClient:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.timeout = 10  # seconds

    def predict_batch(self, csv_data):
        """Sends a batch of loan applications as a CSV string to the prediction API."""
        endpoint = f"{self.base_url}/predict"
        headers = {'Content-Type': 'text/csv'}

        try:
            response = requests.post(endpoint, data=csv_data.encode('utf-8'), headers=headers, timeout=self.timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            logger.info("Successfully received prediction from API.")
            response_data = response.json()
            predictions = response_data.get('predictions')

            if predictions is None:
                logger.error("API response did not contain 'predictions' key.")
                st.error("Received an invalid response from the risk assessment service.")
                return None

            # Re-create DataFrame from the original CSV data to append predictions
            result_df = pd.read_csv(io.StringIO(csv_data))
            result_df['prediction'] = predictions
            return result_df
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} {e.response.reason}")
            try:
                error_data = e.response.json()
                logger.error(f"Structured error from backend: {error_data}")
                error_type = error_data.get('error', 'UnknownError')
                message = error_data.get('message', 'No message provided.')
                details = error_data.get('details', {})
                st.error(f"{error_type}: {message}")
                if details:
                    st.warning(f"Details: {details}")
            except ValueError:
                logger.error(f"Could not parse JSON response. Raw response: {e.response.text}")
                st.error(f"An unexpected error occurred: {e.response.reason}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}", exc_info=True)
            st.error(f"Failed to communicate with the risk assessment service: {e}")
            return None

    def health_check(self):
        """Checks the health of the API."""
        endpoint = f"{self.base_url}/health"
        try:
            response = requests.get(endpoint, timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
