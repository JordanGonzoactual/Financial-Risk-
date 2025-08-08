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

            st.divider()
            st.subheader("Model Health")
            st.caption("Optional diagnostic. Does not affect normal workflow.")
            if st.button("Check Model Performance"):
                try:
                    from .performance_notifications import PerformanceNotificationComponent
                    with st.spinner("Running performance checks..."):
                        results = PerformanceNotificationComponent().check_and_notify()
                except Exception as e:
                    st.warning(f"Performance check could not run: {e}")
                    results = {'overall_status': 'error', 'error': str(e)}
                import datetime
                AppState.set_state('performance_last_results', results)
                AppState.set_state('performance_last_checked_at', datetime.datetime.now())

            last = AppState.get_state('performance_last_results')
            ts = AppState.get_state('performance_last_checked_at')
            if last is not None:
                status = last.get('overall_status')
                st.write(f"Status: {status}")
                if ts:
                    st.caption(f"Last checked: {ts}")
                perf = last.get('model_performance', {})
                lat = last.get('prediction_latency', {})
                if 'r2' in perf and 'rmse' in perf:
                    st.write(f"R² {perf['r2']:.3f}, RMSE {perf['rmse']:.3f}")
                if 'elapsed_ms' in lat:
                    st.write(f"Latency: {lat['elapsed_ms']:.1f} ms")
            else:
                st.caption("Run 'Check Model Performance' to view diagnostics.")

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
