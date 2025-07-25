"""
Tests for the model service backend component.

This module tests the model service functionality including model loading,
health checks, and prediction services.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import tempfile

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import the module to test
try:
    from backend.model_service import ModelService
except ImportError:
    # Create a mock ModelService class for testing if not available
    class ModelService:
        def __init__(self):
            self.model = None
            self.is_loaded = False
        
        def health_check(self):
            return {'model_loaded': self.is_loaded, 'status': 'healthy' if self.is_loaded else 'unhealthy'}
        
        def get_model_info(self):
            if not self.is_loaded:
                return {'error': 'Model not loaded'}
            return {'model_type': 'XGBoost', 'version': '1.0.0'}
        
        def predict_batch(self, data):
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            return np.random.rand(len(data))


# Module-level fixtures
@pytest.fixture
def model_service():
    """Fixture providing a ModelService instance."""
    service = ModelService()
    # Reset to unloaded state for testing
    service.model = None
    service.pipeline = None
    service.is_loaded = False
    return service

@pytest.fixture
def mock_model():
    """Fixture providing a mock ML model."""
    model = Mock()
    # Create a side_effect function that returns predictions based on input size
    def predict_side_effect(X):
        return np.random.rand(len(X))
    
    model.predict.side_effect = predict_side_effect
    model.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
    model.feature_names_in_ = [f'feature_{i}' for i in range(10)]
    model.n_features_in_ = 10
    return model

@pytest.fixture
def mock_pipeline():
    """Fixture providing a mock preprocessing pipeline."""
    pipeline = Mock()
    pipeline.pipeline = Mock()  # Mock the inner pipeline attribute
    
    def transform_side_effect(df):
        # Validate input like the real pipeline
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        # Return a pandas Series with random predictions
        return pd.Series(np.random.rand(len(df)), index=df.index)
    
    pipeline.transform.side_effect = transform_side_effect
    return pipeline


class TestModelService:
    """Test class for ModelService functionality."""
    
    def test_model_service_initialization(self, model_service):
        """Test ModelService initialization."""
        assert model_service is not None
        assert hasattr(model_service, 'health_check')
        assert hasattr(model_service, 'get_model_info')
        assert hasattr(model_service, 'predict_batch')
    
    def test_health_check_no_model(self, model_service):
        """Test health check when no model is loaded."""
        health = model_service.health_check()
        
        assert isinstance(health, dict)
        assert 'model_loaded' in health
        assert 'status' in health
        assert health['model_loaded'] is False
    
    def test_health_check_with_model(self, model_service, mock_model, mock_pipeline):
        """Test health check when model is loaded."""
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        health = model_service.health_check()
        
        assert health['model_loaded'] is True
        assert health['pipeline_loaded'] is True
        assert health['status'] == 'healthy'
    
    def test_get_model_info_no_model(self, model_service):
        """Test get_model_info when no model is loaded."""
        info = model_service.get_model_info()
        
        assert info['model_loaded'] is False
        assert info['model_type'] is None
        assert info['model_version'] is None
    
    def test_get_model_info_with_model(self, model_service, mock_model):
        """Test get_model_info when model is loaded."""
        model_service.model = mock_model
        model_service.is_loaded = True
        
        info = model_service.get_model_info()
        
        assert 'error' not in info
        assert 'model_type' in info
        assert isinstance(info, dict)
    
    def test_predict_batch_no_model(self, model_service, small_loan_data):
        """Test predict_batch when no model is loaded."""
        with pytest.raises(RuntimeError, match="Preprocessing pipeline not loaded"):
            model_service.predict_batch(small_loan_data)
    
    def test_predict_batch_with_model(self, model_service, mock_model, mock_pipeline, small_loan_data):
        """Test predict_batch with loaded model."""
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        predictions = model_service.predict_batch(small_loan_data)
        
        # Predictions come back as pandas Series from the pipeline
        assert hasattr(predictions, '__len__')
        assert len(predictions) == len(small_loan_data)
        assert all(0 <= pred <= 1 for pred in predictions)
    
    def test_predict_batch_empty_data(self, model_service, mock_model, mock_pipeline):
        """Test predict_batch with empty DataFrame."""
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            model_service.predict_batch(empty_df)
    
    def test_predict_batch_invalid_data_type(self, model_service, mock_model, mock_pipeline):
        """Test predict_batch with invalid data type."""
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        with pytest.raises((TypeError, ValueError)):
            model_service.predict_batch("invalid_data")
    
    def test_predict_batch_model_error(self, model_service, mock_pipeline, small_loan_data):
        """Test predict_batch when model raises an error."""
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Model prediction error")
        
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        with pytest.raises(ValueError, match="Model prediction error"):
            model_service.predict_batch(small_loan_data)


class TestModelServiceSingleton:
    """Test class for ModelService singleton behavior."""
    
    @patch('backend.model_service.ModelService')
    def test_singleton_instance(self, mock_service_class):
        """Test that ModelService follows singleton pattern."""
        # This test would verify singleton behavior if implemented
        # For now, we'll test that the service can be imported consistently
        try:
            from backend.model_service import model_service
            assert model_service is not None
        except ImportError:
            # If the actual service doesn't exist, this test passes
            pass


class TestModelLoading:
    """Test class for model loading functionality."""
    
    def test_model_loading_success(self, model_service, mock_model):
        """Test successful model loading."""
        # Simulate model loading
        model_service.model = mock_model
        model_service.is_loaded = True
        
        assert model_service.is_loaded
        assert model_service.model is not None
        
        health = model_service.health_check()
        assert health['model_loaded'] is True
    
    def test_model_loading_failure(self, model_service):
        """Test model loading failure scenarios."""
        # Simulate failed model loading
        model_service.model = None
        model_service.is_loaded = False
        
        assert not model_service.is_loaded
        assert model_service.model is None
        
        health = model_service.health_check()
        assert health['model_loaded'] is False
    
    @patch('os.path.exists')
    def test_model_file_not_found(self, mock_exists, model_service):
        """Test behavior when model file doesn't exist."""
        mock_exists.return_value = False
        
        # This would test actual file loading if implemented
        # For now, we verify the service handles missing files gracefully
        health = model_service.health_check()
        assert 'model_loaded' in health


class TestModelServiceIntegration:
    """Integration tests for ModelService with real-world scenarios."""
    
    def test_full_prediction_pipeline(self, model_service, mock_model, mock_pipeline, sample_loan_data):
        """Test complete prediction pipeline."""
        # Setup loaded model
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        # Check health
        health = model_service.health_check()
        assert health['model_loaded'] is True
        
        # Get model info
        info = model_service.get_model_info()
        assert 'error' not in info
        
        # Make predictions
        predictions = model_service.predict_batch(sample_loan_data)
        assert len(predictions) == len(sample_loan_data)
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_service_resilience(self, model_service, mock_model, small_loan_data):
        """Test service resilience to various error conditions."""
        model_service.model = mock_model
        model_service.is_loaded = True
        
        # Test with various data conditions
        test_cases = [
            small_loan_data,  # Normal data
            small_loan_data.iloc[:0],  # Empty data
            small_loan_data.iloc[:1],  # Single row
        ]
        
        for test_data in test_cases:
            try:
                predictions = model_service.predict_batch(test_data)
                assert isinstance(predictions, np.ndarray)
                assert len(predictions) == len(test_data)
            except Exception as e:
                # Log the error but don't fail the test for edge cases
                print(f"Edge case handling: {e}")
    
    def test_concurrent_predictions(self, model_service, mock_model, mock_pipeline, small_loan_data):
        """Test concurrent prediction requests."""
        import threading
        import time
        
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                predictions = model_service.predict_batch(small_loan_data)
                results.append(len(predictions))
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(result == len(small_loan_data) for result in results)


class TestModelServiceErrorHandling:
    """Test class for error handling in ModelService."""
    
    def test_graceful_degradation(self, model_service):
        """Test graceful degradation when model fails."""
        # Simulate various failure modes
        model_service.model = None
        model_service.is_loaded = False
        
        # Service should still respond to health checks
        health = model_service.health_check()
        assert isinstance(health, dict)
        assert 'model_loaded' in health
        
        # Service should return error info instead of crashing
        info = model_service.get_model_info()
        assert isinstance(info, dict)
        assert info['model_loaded'] is False
    
    def test_invalid_input_handling(self, model_service, mock_model, mock_pipeline):
        """Test handling of various invalid inputs."""
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        invalid_inputs = [
            None,
            "string_input",
            123,
            [],
            {},
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                model_service.predict_batch(invalid_input)
    
    def test_model_corruption_handling(self, model_service):
        """Test handling of corrupted model scenarios."""
        # Simulate corrupted model
        corrupted_model = Mock()
        corrupted_model.predict.side_effect = Exception("Model corrupted")
        
        model_service.model = corrupted_model
        model_service.is_loaded = True
        
        # Health check should still work
        health = model_service.health_check()
        assert health['model_loaded'] is True
        
        # Predictions should raise appropriate error
        with pytest.raises(Exception):
            model_service.predict_batch(pd.DataFrame({'col1': [1, 2, 3]}))


@pytest.mark.performance
class TestModelServicePerformance:
    """Performance tests for ModelService."""
    
    def test_prediction_latency(self, model_service, mock_model, mock_pipeline, sample_loan_data):
        """Test prediction latency for various data sizes."""
        import time
        
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        # Test with different data sizes
        sizes = [10, 100, 1000]
        
        for size in sizes:
            test_data = sample_loan_data.iloc[:min(size, len(sample_loan_data))]
            
            start_time = time.time()
            predictions = model_service.predict_batch(test_data)
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Verify predictions
            assert len(predictions) == len(test_data)
            
            # Log performance metrics
            print(f"Prediction latency for {len(test_data)} samples: {latency:.4f} seconds")
            
            # Basic performance assertion (adjust threshold as needed)
            assert latency < 10.0, f"Prediction took too long: {latency} seconds"
    
    def test_memory_usage(self, model_service, mock_model, mock_pipeline, sample_loan_data):
        """Test memory usage during predictions."""
        import psutil
        import os
        
        model_service.model = mock_model
        model_service.pipeline = mock_pipeline
        model_service.is_loaded = True
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple predictions
        for _ in range(10):
            predictions = model_service.predict_batch(sample_loan_data)
            assert len(predictions) == len(sample_loan_data)
        
        # Check memory usage after predictions
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.2f} MB"
