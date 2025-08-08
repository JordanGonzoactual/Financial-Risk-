import streamlit as st
import pandas as pd
import plotly.express as px
from .state import AppState

# Define required columns for UI display (matches RawData model)
RAW_FEATURE_SCHEMA = [
    'ApplicationDate', 'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus',
    'EducationLevel', 'Experience', 'LoanAmount', 'LoanDuration',
    'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments',
    'CreditCardUtilizationRate', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
    'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose', 'PreviousLoanDefaults',
    'PaymentHistory', 'LengthOfCreditHistory', 'SavingsAccountBalance',
    'CheckingAccountBalance', 'TotalAssets', 'TotalLiabilities', 'MonthlyIncome',
    'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth', 'BaseInterestRate',
    'InterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio'
]

class DataRequirementsComponent:
    """Displays the required data format information for CSV uploads."""
    
    def render(self):
        """Renders the data requirements section at the top of the application."""
        st.header("📋 Data Requirements")
        
        # Main explanation
        st.info(
            "**Important:** All fields listed below must be present in your CSV file for accurate predictions. "
            "Please ensure your data includes all required columns with complete information."
        )
        
        # Handle schema loading gracefully
        if RAW_FEATURE_SCHEMA is None:
            st.error(
                "⚠️ Unable to load data requirements schema. Please contact support if this issue persists."
            )
            st.warning(
                "**Fallback Information:** Your CSV file should contain loan application data with "
                "borrower demographics, financial information, credit history, and loan details."
            )
            return
        
        # Display column count
        total_columns = len(RAW_FEATURE_SCHEMA)
        st.success(f"**Total Required Columns: {total_columns}**")
        
        # Display columns in a 3-column grid for better readability
        st.subheader("Required Column Names:")
        
        # Calculate columns per grid column (roughly equal distribution)
        cols_per_section = (total_columns + 2) // 3  # Round up division
        
        col1, col2, col3 = st.columns(3)
        
        # Use column methods directly (not as context managers)
        col1.markdown("**Section 1:**")
        for i, column in enumerate(RAW_FEATURE_SCHEMA[:cols_per_section]):
            col1.write(f"• {column}")
        
        col2.markdown("**Section 2:**")
        start_idx = cols_per_section
        end_idx = min(cols_per_section * 2, total_columns)
        for column in RAW_FEATURE_SCHEMA[start_idx:end_idx]:
            col2.write(f"• {column}")
        
        col3.markdown("**Section 3:**")
        start_idx = cols_per_section * 2
        for column in RAW_FEATURE_SCHEMA[start_idx:]:
            col3.write(f"• {column}")
        
        # Additional helpful information
        with st.expander("💡 Tips for Data Preparation"):
            st.markdown(
                """
                - **Column Names:** Must match exactly as listed above (case-sensitive)
                - **Data Completeness:** All rows should have values for all columns
                - **File Format:** Upload as CSV or Excel (.xlsx) file
                - **Data Types:** The application will handle type conversion automatically
                - **Missing Values:** Avoid empty cells where possible for best results
                """
            )
        
        # Visual separator
        st.divider()

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
            # Immediately show a small preview after upload for UX and tests
            if AppState.get_state('raw_data') is not None:
                st.dataframe(AppState.get_state('raw_data').head(10))

class DataPreviewComponent:
    """Displays a preview of the uploaded data and handles the validation step."""
    def __init__(self, controller):
        self.controller = controller

    def render(self):
        if AppState.get_state('raw_data') is not None:
            st.header("2. Validate Data")
            # Removed duplicate preview table here (already shown in Step 1)

            with st.expander("View Data Statistics"):
                st.write("**Basic Statistics:**")
                st.dataframe(AppState.get_state('raw_data').describe())
            
            # --- Validation Step ---
            if st.button("Validate Data"):
                with st.spinner('Validating...'):
                    self.controller.handle_data_validation()
                # st.rerun() # Temporarily disabled for debugging

            report = AppState.get_state('validation_report')
            if report:
                self._render_validation_report(report)
                
                # --- Confirmation Step ---
                # Show confirmation button only if data is valid and not yet confirmed
                if report['is_valid'] and not AppState.get_state('data_confirmed'):
                    if st.button("Confirm Validated Data"):
                        AppState.set_state('data_confirmed', True)
                        st.rerun()

            # --- Processing Step ---
            # Show processing controls only after data has been confirmed
            if AppState.get_state('data_confirmed'):
                st.success("Data confirmed and ready for processing.")
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
        # Performance checks are manual-only via the sidebar.

        if st.button("Run Risk Assessment"):
            self.controller.handle_assessment_processing()

class ResultsDashboardComponent:
    """Displays the assessment results in a tabbed dashboard."""
    def render(self):
        results = AppState.get_state('assessment_results')
        if results is not None:
            st.header("4. Assessment Results")
            # Performance checks are manual-only; no automatic checks here.

            # Persist selected section; remove empty Summary option
            labels = ["Detailed Results", "Visualizations"]
            current = AppState.get_state('results_active_tab')
            if current not in labels:
                # Default to Visualizations when prior state was Summary or None
                AppState.set_state('results_active_tab', 'Visualizations')
                current = 'Visualizations'
            default_index = labels.index(current)

            selected = st.radio(
                "View",
                labels,
                index=default_index,
                key="results_active_tab",
            )

            if selected == "Detailed Results":
                st.dataframe(results)
            else:  # Visualizations
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
        # Prefer the predictions column produced by the API client
        score_col = None
        for c in ['prediction', 'risk_score', 'RiskScore']:
            if c in results.columns:
                score_col = c
                break

        if score_col is None:
            st.info("Risk score data not available for visualization.")
            return

        chart_type = st.selectbox("Chart type", ["Box", "Histogram"], index=0, key="risk_chart_type")
        st.caption("Scores are expected to be in the 0–100 range.")

        if chart_type == "Histogram":
            bins = st.slider("Bins", min_value=10, max_value=100, value=30, step=5)
            fig = px.histogram(
                results,
                x=score_col,
                nbins=bins,
                title="Risk Score Distribution",
                labels={score_col: "Risk Score (0–100)"},
            )
            fig.update_layout(xaxis=dict(range=[0, 100]))
        elif chart_type == "Box":
            fig = px.box(
                results,
                y=score_col,
                points="outliers",
                title="Risk Score Box Plot",
                labels={score_col: "Risk Score (0–100)"},
            )
            fig.update_layout(yaxis=dict(range=[0, 100]))
       

        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        try:
            s = results[score_col].astype(float)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean", f"{s.mean():.2f}")
            c2.metric("Median", f"{s.median():.2f}")
            c3.metric("Std Dev", f"{s.std(ddof=0):.2f}")
        except Exception:
            pass

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
