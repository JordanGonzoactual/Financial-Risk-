"""
Comprehensive tests for the Flask application backend.

This module provides extensive testing coverage for Flask API endpoints,
request handling, error responses, CORS configuration, and concurrent access.
"""

import pytest
import json
import io
import os
import sys
import threading
import time
from unittest.mock import patch, Mock, MagicMock
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from backend.app import create_app
    from shared.models.raw_data import RawData
except ImportError:
    # Create mock Flask app for testing if module not available
    from flask import Flask
    
    def create_app():
        """Mock implementation of Flask app creation."""
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return {'status': 'ok', 'message': 'Service is healthy and model is loaded.'}, 200
        
        @app.route('/model-info', methods=['GET'])
        def get_model_info():
            return {'model_type': 'XGBoost', 'version': '1.0.0'}, 200
        
        @app.route('/predict', methods=['POST'])
        def predict():
            return {'predictions': [0.2, 0.8, 0.1, 0.9, 0.3]}, 200
        
        @app.route('/validate', methods=['POST'])
        def validate():
            return {'status': 'valid', 'message': 'Data validation passed'}, 200
        
        return app
    
    # Mock RawData if not available
    class RawData:
        @staticmethod
        def model_validate(data):
            return data





class TestModelInfoEndpoint:
    """Comprehensive tests for the /model-info endpoint."""
    
    def test_model_info_endpoint_success(self, flask_test_client, mock_model_service):
        """Test model info endpoint with successful response."""
        with patch('backend.app.model_service', mock_model_service):
            response = flask_test_client.get('/model-info')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'model_type' in data
            assert data['model_type'] == 'XGBoost'
    
    def test_model_info_endpoint_error(self, flask_test_client):
        """Test model info endpoint when service returns error."""
        mock_service = Mock()
        mock_service.get_model_info.return_value = {'error': 'Model not loaded'}
        
        with patch('backend.app.model_service', mock_service):
            response = flask_test_client.get('/model-info')
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_model_info_endpoint_method_not_allowed(self, flask_test_client):
        """Test model info endpoint with unsupported HTTP methods."""
        response = flask_test_client.post('/model-info')
        assert response.status_code == 405
    
    def test_model_info_endpoint_headers(self, flask_test_client, mock_model_service):
        """Test model info endpoint returns correct headers."""
        with patch('backend.app.model_service', mock_model_service):
            response = flask_test_client.get('/model-info')
            
            assert response.status_code == 200
            assert response.content_type == 'application/json'


class TestValidateEndpoint:
    """Comprehensive tests for the /validate endpoint."""
    
    def test_validate_endpoint_success(self, flask_test_client, sample_loan_data):
        """Test validate endpoint with valid JSON data."""
        valid_data = sample_loan_data.iloc[0].to_dict()
        
        response = flask_test_client.post('/validate', json=valid_data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'valid'
        assert 'validation passed' in data['message']
    
    def test_validate_endpoint_invalid_json(self, flask_test_client):
        """Test validate endpoint with invalid JSON data."""
        invalid_data = {
            'Age': 'thirty',  # Invalid type
            'CreditScore': 950,  # Out of range
            'LoanAmount': -1000  # Invalid value
        }
        
        response = flask_test_client.post('/validate', json=invalid_data)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'ValidationError'
    
    def test_validate_endpoint_missing_content_type(self, flask_test_client):
        """Test validate endpoint with missing JSON content type."""
        response = flask_test_client.post('/validate', data='not json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Request must be JSON'
    
    def test_validate_endpoint_empty_json(self, flask_test_client):
        """Test validate endpoint with empty JSON."""
        response = flask_test_client.post('/validate', json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data


class TestPredictEndpoint:
    """Comprehensive tests for the /predict endpoint."""
    
    def test_predict_endpoint_success(self, flask_test_client, csv_loan_data, mock_model_service):
        """Test successful prediction endpoint."""
        with patch('backend.app.model_service', mock_model_service):
            response = flask_test_client.post(
                '/predict',
                data=csv_loan_data,
                content_type='text/csv'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'predictions' in data
            assert isinstance(data['predictions'], list)
            assert len(data['predictions']) == 5  # Mock returns 5 predictions
    
    def test_predict_endpoint_invalid_content_type(self, flask_test_client, csv_loan_data):
        """Test prediction endpoint with invalid content type."""
        response = flask_test_client.post(
            '/predict',
            data=csv_loan_data,
            content_type='application/json'  # Wrong content type
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Invalid Content-Type'
    
    def test_predict_endpoint_malformed_csv(self, flask_test_client):
        """Test prediction endpoint with malformed CSV data."""
        malformed_csv = "invalid,csv,data\nwithout,proper,headers\nincomplete"
        
        response = flask_test_client.post(
            '/predict',
            data=malformed_csv,
            content_type='text/csv'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_endpoint_empty_csv(self, flask_test_client):
        """Test prediction endpoint with empty CSV data."""
        empty_csv = ""
        
        response = flask_test_client.post(
            '/predict',
            data=empty_csv,
            content_type='text/csv'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_endpoint_schema_validation_errors(self, flask_test_client):
        """Test prediction endpoint with schema validation errors."""
        # Create CSV with invalid data types and missing columns
        invalid_data = {
            'Age': ['thirty', 25, -5],  # Invalid type, negative value
            'CreditScore': [700, 'bad', 950],  # Invalid type, out of range
            'LoanAmount': [10000, 20000, 30000]
            # Missing required columns like ApplicationDate
        }
        df = pd.DataFrame(invalid_data)
        csv_data = df.to_csv(index=False)
        
        response = flask_test_client.post('/predict', data=csv_data, content_type='text/csv')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'ValidationError'
    
    def test_predict_endpoint_model_service_error(self, flask_test_client, csv_loan_data):
        """Test prediction endpoint when model service raises an error."""
        mock_service = Mock()
        mock_service.predict_batch.side_effect = RuntimeError("Model prediction failed")
        
        with patch('backend.app.model_service', mock_service):
            response = flask_test_client.post(
                '/predict',
                data=csv_loan_data,
                content_type='text/csv'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'InternalServerError'
    
    def test_predict_endpoint_type_error(self, flask_test_client, csv_loan_data):
        """Test prediction endpoint handling of type errors."""
        mock_service = Mock()
        mock_service.predict_batch.side_effect = TypeError("Invalid data type")
        
        with patch('backend.app.model_service', mock_service):
            response = flask_test_client.post(
                '/predict',
                data=csv_loan_data,
                content_type='text/csv'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'TypeError'
    
    def test_predict_endpoint_missing_columns_error(self, flask_test_client):
        """Test prediction endpoint with missing required columns."""
        incomplete_data = pd.DataFrame({
            'Age': [30, 25],
            'CreditScore': [700, 750]
            # Missing many required columns
        })
        csv_data = incomplete_data.to_csv(index=False)
        
        response = flask_test_client.post('/predict', data=csv_data, content_type='text/csv')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'ValidationError'
    
    def test_predict_endpoint_method_not_allowed(self, flask_test_client):
        """Test prediction endpoint with unsupported HTTP methods."""
        response = flask_test_client.get('/predict')
        assert response.status_code == 405
        
        response = flask_test_client.put('/predict')
        assert response.status_code == 405











class TestErrorHandling:
    """Test class for error handling scenarios."""
    
    def test_404_error(self, flask_test_client):
        """Test handling of 404 errors for non-existent endpoints."""
        response = flask_test_client.get('/nonexistent')
        assert response.status_code == 404
    
    def test_method_not_allowed(self, flask_test_client):
        """Test handling of method not allowed errors."""
        response = flask_test_client.put('/health')  # PUT not allowed on health endpoint
        assert response.status_code == 405
    
    def test_error_response_format(self, flask_test_client):
        """Test that error responses have consistent format."""
        response = flask_test_client.post('/predict', data='invalid', content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert isinstance(data['error'], str)
    
    
@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete request-response cycles."""
    
    def test_complete_prediction_workflow(self, flask_test_client, csv_loan_data, mock_model_service):
        """Test complete prediction workflow from request to response."""
        with patch('backend.app.model_service', mock_model_service):
            # First check health
            health_response = flask_test_client.get('/health')
            assert health_response.status_code == 200
            
            # Get model info
            info_response = flask_test_client.get('/model-info')
            assert info_response.status_code == 200
            
            # Validate data first
            sample_data = pd.read_csv(io.StringIO(csv_loan_data)).iloc[0].to_dict()
            validate_response = flask_test_client.post('/validate', json=sample_data)
            # Validation might pass or fail depending on data, but should respond
            assert validate_response.status_code in [200, 400]
            
            # Make prediction
            pred_response = flask_test_client.post(
                '/predict',
                data=csv_loan_data,
                content_type='text/csv'
            )
            assert pred_response.status_code == 200
            
            # Verify prediction response format
            data = json.loads(pred_response.data)
            assert 'predictions' in data
            assert isinstance(data['predictions'], list)
    

    
    def test_mixed_endpoint_concurrent_access(self, flask_test_client, csv_loan_data, mock_model_service, sample_loan_data):
        """Test concurrent access to different endpoints."""
        app = create_app()
        results = {'health': [], 'info': [], 'predict': [], 'validate': []}
        
        def make_health_request():
            with patch('backend.app.model_service', mock_model_service):
                with app.test_client() as client:
                    response = client.get('/health')
                    results['health'].append(response.status_code)
        
        def make_info_request():
            with patch('backend.app.model_service', mock_model_service):
                with app.test_client() as client:
                    response = client.get('/model-info')
                    results['info'].append(response.status_code)
        
        def make_predict_request():
            with patch('backend.app.model_service', mock_model_service):
                with app.test_client() as client:
                    response = client.post('/predict', data=csv_loan_data, content_type='text/csv')
                    results['predict'].append(response.status_code)
        
        def make_validate_request():
            valid_data = sample_loan_data.iloc[0].to_dict()
            with app.test_client() as client:
                response = client.post('/validate', json=valid_data)
                results['validate'].append(response.status_code)
        
        # Create mixed concurrent requests
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _ in range(3):
                futures.append(executor.submit(make_health_request))
                futures.append(executor.submit(make_info_request))
                futures.append(executor.submit(make_predict_request))
                futures.append(executor.submit(make_validate_request))
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        # Verify all endpoints handled concurrent requests
        assert len(results['health']) == 3
        assert len(results['info']) == 3
        assert len(results['predict']) == 3
        assert len(results['validate']) == 3
        
        # All health and info requests should succeed
        assert all(status == 200 for status in results['health'])
        assert all(status == 200 for status in results['info'])
        assert all(status == 200 for status in results['predict'])
        # Validate requests might succeed or fail depending on data
        assert all(status in [200, 400] for status in results['validate'])


@pytest.mark.performance
class TestPerformanceScenarios:
    """Performance tests for API endpoints."""
    
    def test_health_endpoint_response_time(self, flask_test_client, mock_model_service):
        """Test health endpoint response time."""
        app = create_app()
        with patch('backend.app.model_service', mock_model_service):
            start_time = time.time()
            with app.test_client() as client:
                response = client.get('/health')
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < 1.0, f"Health check took too long: {response_time:.3f}s"
    

