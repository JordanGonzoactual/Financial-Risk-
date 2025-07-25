import streamlit as st
import pandas as pd
import plotly.express as px
from .state import AppState

class FileUploadComponent:
    """Handles the file upload and initial preview."""
    def __init__(self, controller):
        self.controller = controller

    def render(self):
        st.header("1. Upload Loan Application Data")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload your CSV or Excel file.",
            type=["csv", "xlsx"],
            help="Upload a file with the required columns for risk assessment."
        )
        if uploaded_file:
            # Use the controller to handle the file, which also manages state
            self.controller.handle_file_upload(uploaded_file)

class DataPreviewComponent:
    """Displays a preview of the uploaded data and handles the validation step."""
    def __init__(self, controller):
        self.controller = controller

    def render(self):
        if AppState.get_state('raw_data') is not None:
            st.header("2. Validate Data")
            st.dataframe(AppState.get_state('raw_data').head(10))

            with st.expander("View Data Statistics"):
                st.write("**Basic Statistics:**")
                st.dataframe(AppState.get_state('raw_data').describe())
            
            if st.button("Confirm and Validate Data"):
                self.controller.handle_data_validation()

            report = AppState.get_state('validation_report')
            if report:
                self._render_validation_report(report)

            # Show processing button only if data is validated
            if AppState.get_state('validated_data') is not None:
                ProcessingControlsComponent(self.controller).render()

    def _render_validation_report(self, report):
        if not report['is_valid']:
            st.error("Data validation failed. Please correct the errors and re-upload.")
            with st.expander("View Error Details"):
                for error in report['errors']:
                    st.write(error)
        else:
            st.success("Data is valid and clean! Ready for processing.")

        if report['warnings']:
            st.warning("Please review the following warnings:")
            with st.expander("View Warning Details"):
                for warning in report['warnings']:
                    st.write(warning)

class ProcessingControlsComponent:
    """Renders controls for running the risk assessment."""
    def __init__(self, controller):
        self.controller = controller

    def render(self):
        st.header("3. Process for Risk Assessment")
        if st.button("Run Risk Assessment"):
            self.controller.handle_assessment_processing()

class ResultsDashboardComponent:
    """Displays the assessment results in a tabbed dashboard."""
    def render(self):
        results = AppState.get_state('assessment_results')
        if results is not None:
            st.header("4. Assessment Results")
            tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Results", "Visualizations"])

            with tab1:
                self._render_summary(results)
            with tab2:
                st.dataframe(results)
            with tab3:
                self._render_visualizations(results)

    def _render_summary(self, results):
        st.subheader("Risk Assessment Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applicants", f"{len(results)}")
        # Example metrics - these would be based on actual API output
        if 'risk_category' in results.columns:
            avg_risk = results['risk_category'].value_counts().get('High', 0)
            col2.metric("High-Risk Applicants", f"{avg_risk}")
        if 'loan_amount' in results.columns:
            total_loan_amount = results['loan_amount'].sum()
            col3.metric("Total Loan Amount", f"${total_loan_amount:,.2f}")

    def _render_visualizations(self, results):
        st.subheader("Risk Score Distribution")
        if 'risk_score' in results.columns:
            fig = px.histogram(results, x='risk_score', nbins=20, title="Distribution of Risk Scores")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk score data not available for visualization.")

class DownloadManagerComponent:
    """Provides options to download the results."""
    def render(self):
        results = AppState.get_state('assessment_results')
        if results is not None:
            st.subheader("Export Results")
            # CSV Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name='risk_assessment_results.csv',
                mime='text/csv',
            )
