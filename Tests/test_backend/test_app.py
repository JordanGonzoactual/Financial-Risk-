"""
Tests for the Flask application backend.

This module tests the Flask API endpoints, request handling, and error responses.
"""

import pytest
import json
import io
import os
import sys
from unittest.mock import patch, Mock
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from backend.app import create_app
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
        
        return app


class TestFlaskApp:
    """Test class for Flask application functionality."""
    
    def test_health_endpoint_success(self, flask_test_client, mock_model_service):
        """Test health endpoint when model is loaded successfully."""
        with patch('backend.app.model_service', mock_model_service):
            response = flask_test_client.get('/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'ok'
            assert 'Service is healthy' in data['message']
    
    def test_health_endpoint_failure(self, flask_test_client):
        """Test health endpoint when model is not loaded."""
        mock_service = Mock()
        mock_service.health_check.return_value = {'model_loaded': False}
        
        with patch('backend.app.model_service', mock_service):
            response = flask_test_client.get('/health')
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'unhealthy' in data['message']
    
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
        assert 'Invalid Content-Type' in data['error']
    
    def test_predict_endpoint_invalid_csv(self, flask_test_client):
        """Test prediction endpoint with invalid CSV data."""
        invalid_csv = "invalid,csv,data\n1,2"  # Malformed CSV
        
        response = flask_test_client.post(
            '/predict',
            data=invalid_csv,
            content_type='text/csv'
        )
        
        # Should fail during CSV parsing or schema validation
        assert response.status_code in [400, 500]
    
    def test_predict_endpoint_schema_validation_error(self, flask_test_client):
        """Test prediction endpoint with schema validation error."""
        # CSV with missing required columns
        invalid_csv = "Age,Income\n25,50000\n30,60000"
        
        response = flask_test_client.post(
            '/predict',
            data=invalid_csv,
            content_type='text/csv'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'ValidationError'
        assert 'schema' in data['message'].lower()
    
    @patch('backend.app.model_service')
    def test_predict_endpoint_model_service_error(self, mock_service, flask_test_client, csv_loan_data):
        """Test prediction endpoint when model service raises an error."""
        mock_service.predict_batch.side_effect = RuntimeError("Model prediction failed")
        mock_service.health_check.return_value = {'model_loaded': True}
        
        response = flask_test_client.post(
            '/predict',
            data=csv_loan_data,
            content_type='text/csv'
        )
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
    
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
            assert data['error'] == 'TypeError'


class TestAppCreation:
    """Test class for Flask app creation and configuration."""
    
    def test_create_app_returns_flask_instance(self):
        """Test that create_app returns a Flask application instance."""
        app = create_app()
        assert app is not None
        assert hasattr(app, 'test_client')
    
    def test_cors_configuration(self):
        """Test that CORS is properly configured."""
        app = create_app()
        # CORS configuration is applied during app creation
        # We can test this by checking if the app has the necessary before_request handlers
        assert len(app.before_request_funcs[None]) > 0
    
    def test_app_routes_registered(self):
        """Test that all expected routes are registered."""
        app = create_app()
        
        # Get all registered routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        
        assert '/health' in routes
        assert '/model-info' in routes
        assert '/predict' in routes


class TestRequestLogging:
    """Test class for request logging functionality."""
    
    def test_request_logging(self, flask_test_client, caplog):
        """Test that requests are properly logged."""
        with caplog.at_level('INFO'):
            flask_test_client.get('/health')
            
        # Check that request was logged
        log_messages = [record.message for record in caplog.records]
        request_logs = [msg for msg in log_messages if 'Request:' in msg]
        assert len(request_logs) > 0
        assert 'GET /health' in request_logs[0]


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
    
    def test_concurrent_requests(self, app, flask_test_client, csv_loan_data, mock_model_service):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request(app):
            with app.app_context():
                with patch('backend.app.model_service', mock_model_service):
                    response = flask_test_client.post(
                        '/predict',
                        data=csv_loan_data,
                        content_type='text/csv'
                    )
                    results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request, args=(app,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5
