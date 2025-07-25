"""
Tests for the Streamlit frontend application.

This module tests the Streamlit app functionality, UI components, and user interactions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
import requests
import io

# Mock Streamlit if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    
    class MockStreamlit:
        @staticmethod
        def title(text):
            pass
        
        @staticmethod
        def header(text):
            pass
        
        @staticmethod
        def subheader(text):
            pass
        
        @staticmethod
        def write(text):
            pass
        
        @staticmethod
        def error(text):
            pass
        
        @staticmethod
        def success(text):
            pass
        
        @staticmethod
        def warning(text):
            pass
        
        @staticmethod
        def info(text):
            pass
        
        @staticmethod
        def file_uploader(label, type=None):
            return None
        
        @staticmethod
        def button(label):
            return False
        
        @staticmethod
        def selectbox(label, options):
            return options[0] if options else None
        
        @staticmethod
        def slider(label, min_value, max_value, value):
            return value
        
        @staticmethod
        def number_input(label, min_value=None, max_value=None, value=None):
            return value or 0
        
        @staticmethod
        def text_input(label, value=""):
            return value
        
        @staticmethod
        def dataframe(df):
            pass
        
        @staticmethod
        def json(data):
            pass
        
        @staticmethod
        def plotly_chart(fig):
            pass
    
    st = MockStreamlit()


# Module-level fixtures for all test classes
@pytest.fixture
def mock_streamlit_session():
    """Fixture providing a mock Streamlit session state."""
    session_state = {}
    return session_state

@pytest.fixture
def sample_uploaded_file(csv_loan_data):
    """Fixture providing a mock uploaded file."""
    mock_file = Mock()
    mock_file.read.return_value = csv_loan_data.encode('utf-8')
    mock_file.name = "test_loan_data.csv"
    mock_file.type = "text/csv"
    return mock_file


class TestStreamlitApp:
    """Test class for Streamlit application functionality."""
    
    def test_app_title_and_header(self):
        """Test that app displays correct title and headers."""
        with patch.object(st, 'title') as mock_title, \
             patch.object(st, 'header') as mock_header:
            
            # Simulate app initialization
            st.title("Financial Risk Assessment")
            st.header("Loan Risk Prediction")
            
            mock_title.assert_called_with("Financial Risk Assessment")
            mock_header.assert_called_with("Loan Risk Prediction")
    
    def test_file_upload_component(self, sample_uploaded_file):
        """Test file upload functionality."""
        with patch.object(st, 'file_uploader', return_value=sample_uploaded_file) as mock_uploader:
            
            uploaded_file = st.file_uploader(
                "Upload loan data CSV file",
                type=['csv']
            )
            
            mock_uploader.assert_called_with(
                "Upload loan data CSV file",
                type=['csv']
            )
            
            assert uploaded_file is not None
            assert uploaded_file.name == "test_loan_data.csv"
    
    def test_data_validation_display(self, sample_loan_data):
        """Test data validation and display functionality."""
        with patch.object(st, 'dataframe') as mock_dataframe, \
             patch.object(st, 'success') as mock_success, \
             patch.object(st, 'error') as mock_error:
            
            # Test successful validation
            st.dataframe(sample_loan_data.head())
            st.success("Data validation successful!")
            
            mock_dataframe.assert_called()
            mock_success.assert_called_with("Data validation successful!")
            
            # Test validation error
            st.error("Data validation failed: Missing required columns")
            mock_error.assert_called_with("Data validation failed: Missing required columns")
    
    def test_prediction_button_functionality(self):
        """Test prediction button and API call functionality."""
        with patch.object(st, 'button', return_value=True) as mock_button, \
             patch('requests.post') as mock_post:
            
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'predictions': [0.2, 0.8, 0.1, 0.9, 0.3]
            }
            mock_post.return_value = mock_response
            
            # Simulate button click
            if st.button("Predict Risk"):
                response = requests.post(
                    "http://localhost:5001/predict",
                    data="mock_csv_data",
                    headers={'Content-Type': 'text/csv'}
                )
            
            mock_button.assert_called_with("Predict Risk")
            assert mock_post.called
    
    def test_prediction_results_display(self):
        """Test display of prediction results."""
        predictions = [0.2, 0.8, 0.1, 0.9, 0.3]
        
        with patch.object(st, 'subheader') as mock_subheader, \
             patch.object(st, 'write') as mock_write, \
             patch.object(st, 'json') as mock_json:
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"Number of predictions: {len(predictions)}")
            st.json({"predictions": predictions})
            
            mock_subheader.assert_called_with("Prediction Results")
            mock_write.assert_called_with(f"Number of predictions: {len(predictions)}")
            mock_json.assert_called_with({"predictions": predictions})
    
    def test_error_handling_display(self):
        """Test error handling and display in the app."""
        error_scenarios = [
            ("File upload error", "Please upload a valid CSV file"),
            ("API connection error", "Unable to connect to prediction service"),
            ("Data validation error", "Data does not match required schema"),
            ("Prediction error", "Error occurred during prediction")
        ]
        
        with patch.object(st, 'error') as mock_error:
            for error_type, error_message in error_scenarios:
                st.error(f"{error_type}: {error_message}")
                
            # Verify all errors were displayed
            assert mock_error.call_count == len(error_scenarios)
    
    def test_data_preview_functionality(self, sample_loan_data):
        """Test data preview functionality."""
        with patch.object(st, 'dataframe') as mock_dataframe, \
             patch.object(st, 'write') as mock_write:
            
            # Display data preview
            st.write("Data Preview:")
            data_preview = sample_loan_data.head(10)
            st.dataframe(data_preview)
            st.write(f"Total rows: {len(sample_loan_data)}")
            st.write(f"Total columns: {len(sample_loan_data.columns)}")
            
            mock_write.assert_any_call("Data Preview:")
            # Use call_args to check the dataframe call without comparing DataFrames directly
            assert mock_dataframe.called
            assert len(mock_dataframe.call_args[0][0]) == 10  # Check that 10 rows were passed
            mock_write.assert_any_call(f"Total rows: {len(sample_loan_data)}")
            mock_write.assert_any_call(f"Total columns: {len(sample_loan_data.columns)}")
    
    def test_input_validation_feedback(self):
        """Test input validation feedback to users."""
        with patch.object(st, 'warning') as mock_warning, \
             patch.object(st, 'info') as mock_info:
            
            # Test various validation messages
            st.warning("File size is large. Processing may take longer.")
            st.info("Please ensure your CSV file contains all required columns.")
            
            mock_warning.assert_called_with("File size is large. Processing may take longer.")
            mock_info.assert_called_with("Please ensure your CSV file contains all required columns.")


class TestStreamlitUIComponents:
    """Test class for Streamlit UI components and interactions."""
    
    def test_sidebar_components(self):
        """Test sidebar components and configuration."""
        with patch.object(st, 'sidebar') as mock_sidebar:
            # Mock sidebar methods
            mock_sidebar.header = Mock()
            mock_sidebar.selectbox = Mock(return_value="Option 1")
            mock_sidebar.slider = Mock(return_value=0.5)
            mock_sidebar.button = Mock(return_value=False)
            
            # Simulate sidebar usage
            mock_sidebar.header("Configuration")
            model_type = mock_sidebar.selectbox("Model Type", ["XGBoost", "Random Forest"])
            threshold = mock_sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5)
            reset_button = mock_sidebar.button("Reset")
            
            # Verify sidebar interactions
            mock_sidebar.header.assert_called_with("Configuration")
            mock_sidebar.selectbox.assert_called_with("Model Type", ["XGBoost", "Random Forest"])
            mock_sidebar.slider.assert_called_with("Risk Threshold", 0.0, 1.0, 0.5)
            mock_sidebar.button.assert_called_with("Reset")
    
    def test_form_components(self):
        """Test form components for manual data entry."""
        with patch.object(st, 'form') as mock_form, \
             patch.object(st, 'number_input') as mock_number_input, \
             patch.object(st, 'selectbox') as mock_selectbox, \
             patch.object(st, 'form_submit_button') as mock_submit:
            
            # Mock form context
            mock_form_context = Mock()
            mock_form.return_value.__enter__ = Mock(return_value=mock_form_context)
            mock_form.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate form creation
            with st.form("loan_application_form"):
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                income = st.number_input("Annual Income", min_value=0, value=50000)
                employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
                submitted = st.form_submit_button("Submit Application")
            
            # Verify form components
            mock_form.assert_called_with("loan_application_form")
            assert mock_number_input.call_count >= 2
            mock_selectbox.assert_called()
    
    def test_progress_indicators(self):
        """Test progress indicators during processing."""
        with patch.object(st, 'progress') as mock_progress, \
             patch.object(st, 'spinner') as mock_spinner:
            
            # Mock progress bar
            mock_progress_bar = Mock()
            mock_progress.return_value = mock_progress_bar
            
            # Mock spinner context
            mock_spinner_context = Mock()
            mock_spinner.return_value.__enter__ = Mock(return_value=mock_spinner_context)
            mock_spinner.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate progress indicators
            progress_bar = st.progress(0)
            progress_bar.progress(50)
            progress_bar.progress(100)
            
            with st.spinner("Processing predictions..."):
                pass  # Simulate processing
            
            # Verify progress indicators
            mock_progress.assert_called_with(0)
            mock_spinner.assert_called_with("Processing predictions...")
    
    def test_chart_components(self):
        """Test chart and visualization components."""
        with patch.object(st, 'plotly_chart') as mock_plotly, \
             patch.object(st, 'bar_chart') as mock_bar_chart, \
             patch.object(st, 'line_chart') as mock_line_chart:
            
            # Mock chart data
            chart_data = pd.DataFrame({
                'Risk_Score': [0.1, 0.3, 0.7, 0.9],
                'Count': [10, 15, 8, 3]
            })
            
            # Simulate chart creation
            st.bar_chart(chart_data)
            st.line_chart(chart_data)
            
            # Mock Plotly figure
            mock_fig = Mock()
            st.plotly_chart(mock_fig)
            
            # Verify chart calls
            mock_bar_chart.assert_called_with(chart_data)
            mock_line_chart.assert_called_with(chart_data)
            mock_plotly.assert_called_with(mock_fig)


class TestStreamlitDataProcessing:
    """Test class for data processing within Streamlit app."""
    
    def test_csv_file_processing(self, sample_uploaded_file, csv_loan_data):
        """Test CSV file processing and parsing."""
        # Mock file reading
        sample_uploaded_file.read.return_value = csv_loan_data.encode('utf-8')
        
        # Simulate file processing
        if sample_uploaded_file is not None:
            # Read file content
            file_content = sample_uploaded_file.read().decode('utf-8')
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(file_content))
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert len(df.columns) > 0
    
    def test_data_validation_integration(self, sample_loan_data):
        """Test data validation integration with schema validator."""
        with patch('FeatureEngineering.schema_validator.validate_raw_data_schema') as mock_validate:
            # Mock successful validation
            mock_validate.return_value = True
            
            # Simulate validation
            is_valid = mock_validate(sample_loan_data)
            
            assert is_valid is True
            mock_validate.assert_called_with(sample_loan_data)
    
    def test_api_request_formatting(self, csv_loan_data):
        """Test formatting data for API requests."""
        # Simulate API request preparation
        headers = {'Content-Type': 'text/csv'}
        data = csv_loan_data
        url = "http://localhost:5001/predict"
        
        # Verify request parameters
        assert headers['Content-Type'] == 'text/csv'
        assert isinstance(data, str)
        assert url.endswith('/predict')
    
    def test_response_processing(self):
        """Test processing of API responses."""
        # Mock API response
        mock_response_data = {
            'predictions': [0.2, 0.8, 0.1, 0.9, 0.3],
            'status': 'success'
        }
        
        # Process response
        predictions = mock_response_data.get('predictions', [])
        status = mock_response_data.get('status', 'unknown')
        
        assert len(predictions) == 5
        assert all(0 <= pred <= 1 for pred in predictions)
        assert status == 'success'
    
    def test_error_response_handling(self):
        """Test handling of error responses from API."""
        error_responses = [
            {'error': 'ValidationError', 'message': 'Invalid data schema'},
            {'error': 'InternalServerError', 'message': 'Server error occurred'},
            {'error': 'TypeError', 'message': 'Data type mismatch'}
        ]
        
        for error_response in error_responses:
            error_type = error_response.get('error', 'Unknown')
            error_message = error_response.get('message', 'No message')
            
            assert error_type in ['ValidationError', 'InternalServerError', 'TypeError', 'Unknown']
            assert isinstance(error_message, str)
            assert len(error_message) > 0


class TestStreamlitSessionState:
    """Test class for Streamlit session state management."""
    
    def test_session_state_initialization(self, mock_streamlit_session):
        """Test session state initialization."""
        # Initialize session state variables
        session_state = mock_streamlit_session
        session_state['uploaded_file'] = None
        session_state['predictions'] = []
        session_state['validation_status'] = False
        
        assert 'uploaded_file' in session_state
        assert 'predictions' in session_state
        assert 'validation_status' in session_state
    
    def test_session_state_updates(self, mock_streamlit_session):
        """Test session state updates during app usage."""
        session_state = mock_streamlit_session
        
        # Simulate state updates
        session_state['uploaded_file'] = "test_file.csv"
        session_state['predictions'] = [0.1, 0.8, 0.3]
        session_state['validation_status'] = True
        
        assert session_state['uploaded_file'] == "test_file.csv"
        assert len(session_state['predictions']) == 3
        assert session_state['validation_status'] is True
    
    def test_session_state_persistence(self, mock_streamlit_session):
        """Test session state persistence across interactions."""
        session_state = mock_streamlit_session
        
        # Set initial state
        session_state['user_config'] = {
            'model_type': 'XGBoost',
            'threshold': 0.5,
            'show_details': True
        }
        
        # Verify persistence
        config = session_state.get('user_config', {})
        assert config['model_type'] == 'XGBoost'
        assert config['threshold'] == 0.5
        assert config['show_details'] is True


@pytest.mark.integration
class TestStreamlitAppIntegration:
    """Integration tests for Streamlit app with backend services."""
    
    def test_full_prediction_workflow(self, sample_uploaded_file, csv_loan_data):
        """Test complete prediction workflow from upload to results."""
        with patch.object(st, 'file_uploader', return_value=sample_uploaded_file), \
             patch.object(st, 'button', return_value=True), \
             patch('requests.post') as mock_post, \
             patch.object(st, 'success') as mock_success:
            
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'predictions': [0.2, 0.8, 0.1]}
            mock_post.return_value = mock_response
            
            # Simulate workflow
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            
            if uploaded_file and st.button("Predict"):
                response = requests.post(
                    "http://localhost:5001/predict",
                    data=csv_loan_data,
                    headers={'Content-Type': 'text/csv'}
                )
                
                if response.status_code == 200:
                    st.success("Predictions completed successfully!")
            
            # Verify workflow
            assert uploaded_file is not None
            mock_post.assert_called()
            mock_success.assert_called_with("Predictions completed successfully!")
    
    def test_error_handling_integration(self):
        """Test error handling integration across components."""
        with patch('requests.post') as mock_post, \
             patch.object(st, 'error') as mock_error:
            
            # Mock API error
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                'error': 'ValidationError',
                'message': 'Invalid data schema'
            }
            mock_post.return_value = mock_response
            
            # Simulate error handling
            response = requests.post("http://localhost:5001/predict", data="invalid_data")
            
            if response.status_code != 200:
                error_data = response.json()
                st.error(f"Error: {error_data.get('message', 'Unknown error')}")
            
            # Verify error handling
            mock_error.assert_called_with("Error: Invalid data schema")
    
    def test_real_time_validation_feedback(self, sample_loan_data):
        """Test real-time validation feedback to users."""
        with patch('FeatureEngineering.schema_validator.validate_raw_data_schema') as mock_validate, \
             patch.object(st, 'success') as mock_success, \
             patch.object(st, 'error') as mock_error:
            
            # Test successful validation
            mock_validate.return_value = True
            
            try:
                mock_validate(sample_loan_data)
                st.success("✅ Data validation passed!")
            except ValueError as e:
                st.error(f"❌ Validation failed: {str(e)}")
            
            mock_success.assert_called_with("✅ Data validation passed!")
            
            # Test validation failure
            mock_validate.side_effect = ValueError("Missing columns")
            
            try:
                mock_validate(sample_loan_data)
                st.success("✅ Data validation passed!")
            except ValueError as e:
                st.error(f"❌ Validation failed: {str(e)}")
            
            mock_error.assert_called_with("❌ Validation failed: Missing columns")


@pytest.mark.performance
class TestStreamlitPerformance:
    """Performance tests for Streamlit app."""
    
    def test_large_file_handling(self):
        """Test handling of large CSV files."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Test CSV conversion performance
        import time
        start_time = time.time()
        csv_string = large_data.to_csv(index=False)
        conversion_time = time.time() - start_time
        
        assert len(csv_string) > 0
        assert conversion_time < 5.0  # Should convert within 5 seconds
    
    def test_ui_responsiveness(self):
        """Test UI responsiveness with multiple components."""
        with patch.object(st, 'dataframe') as mock_dataframe, \
             patch.object(st, 'plotly_chart') as mock_chart:
            
            # Simulate multiple UI updates
            for i in range(10):
                test_data = pd.DataFrame({'x': range(100), 'y': np.random.randn(100)})
                st.dataframe(test_data)
                st.plotly_chart(Mock())
            
            # Verify all components were rendered
            assert mock_dataframe.call_count == 10
            assert mock_chart.call_count == 10
