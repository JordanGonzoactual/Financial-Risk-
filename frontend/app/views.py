import streamlit as st
from .state import AppState
from .controller import AppController
from .ui_components import (
    DataRequirementsComponent,
    FileUploadComponent,
    DataPreviewComponent,
    ResultsDashboardComponent,
    DownloadManagerComponent
)

class AppViews:
    def __init__(self):
        self.controller = AppController()
        # Instantiate components
        self.data_requirements = DataRequirementsComponent()
        self.file_uploader = FileUploadComponent(self.controller)
        self.data_preview = DataPreviewComponent(self.controller) # Pass controller for validation button
        self.results_dashboard = ResultsDashboardComponent()
        self.download_manager = DownloadManagerComponent()

    def render_sidebar(self):
        with st.sidebar:
            st.header("Configuration")
            st.info("The application is currently configured to connect to the Financial Risk API. System status and endpoint details will be displayed here.")

    def render_main_content(self):
        st.title("Batch Loan Risk Assessment")

        # Display data requirements first
        self.data_requirements.render()
        
        # Render components in order
        self.file_uploader.render()
        self.data_preview.render()

        # The processing button is now part of the data preview component's flow
        # and the results dashboard handles the rest
        self.results_dashboard.render()

        # The download manager also checks for results internally
        if AppState.get_state('assessment_results') is not None:
            self.download_manager.render()
