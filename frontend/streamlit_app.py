import sys
import os

# Add the frontend directory to the Python path to resolve module import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from app.views import AppViews
from app.controller import AppController
from app.state import AppState
from utils.logging import get_logger
from utils.backend_launcher import start_flask_backend

logger = get_logger(__name__)

class RiskAssessmentApp:
    def __init__(self, views, controller):
        self.views = views
        self.controller = controller
        self.initialized = self._initialize_app()

    def _initialize_app(self):
        st.set_page_config(
            page_title="Financial Risk Assessment",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._load_css("assets/style.css")

        with st.spinner("Starting backend services..."):
            backend_status = start_flask_backend()
            if backend_status is not True:
                # If start_flask_backend returns a string, it's an error message
                error_msg = backend_status if isinstance(backend_status, str) else "Failed to start Flask backend. Please check the logs for details."
                st.error(error_msg)
                logger.error(error_msg)
                return False
        
        AppState.initialize()
        return True

    def _load_css(self, file_name):
        try:
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            # During tests, suppress warning for the default CSS to reduce noise
            try:
                import os
                if os.environ.get("PYTEST_CURRENT_TEST") and os.path.basename(file_name) == "style.css":
                    return
            except Exception:
                pass
            logger.warning(f"CSS file not found: {file_name}")

    def run(self):
        if not self.initialized:
            logger.warning("Application not initialized, skipping run.")
            return

        try:
            self.views.render_sidebar()
            self.views.render_main_content()
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            try:
                # Attempt to parse the structured error response from the backend
                error_json = e.response.json()
                category = error_json.get('category', 'generic')
                details = error_json.get('details', 'No details provided.')

                if category == 'missing_columns':
                    missing_cols = error_json.get('missing_columns', [])
                    st.error(f"Data Validation Error: Your CSV is missing the following required columns: {', '.join(missing_cols)}. Please correct the file and try again.")
                else:
                    st.error(f"An error occurred: {details}")
            except Exception as parse_error:
                # Fallback for non-structured errors
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    logger.info("Application starting...")
    views = AppViews()
    controller = AppController()
    app = RiskAssessmentApp(views, controller)
    app.run()
    logger.info("Application finished.")
