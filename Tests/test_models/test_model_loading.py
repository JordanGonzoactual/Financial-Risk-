import os
import pytest
from pathlib import Path
from backend.model_service import ModelService

def test_singleton_initialization():
    service1 = ModelService()
    service2 = ModelService()
    assert service1 is service2, "ModelService should be a singleton"

def test_model_file_loading():
    service = ModelService()
    model_path = Path(__file__).parents[2] / 'Models' / 'trained_models' / 'final_model.pkl'
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert hasattr(service, 'model'), "ModelService missing 'model' attribute"
    assert service.model is not None, "Model should be loaded"

def test_preprocessing_pipeline_loading():
    service = ModelService()
    pipeline_path = Path(__file__).parents[2] / 'FeatureEngineering' / 'artifacts'
    assert pipeline_path.exists(), f"Pipeline artifacts not found at {pipeline_path}"
    assert hasattr(service, 'pipeline'), "ModelService missing 'pipeline' attribute"
    assert service.pipeline is not None, "Preprocessing pipeline should be loaded"

def test_health_check():
    service = ModelService()
    health = service.health_check()
    assert isinstance(health, dict)
    assert health.get('status') == 'healthy'
    assert health.get('model_loaded') is True
    assert health.get('pipeline_loaded') is True

def test_missing_model_file(monkeypatch, tmp_path):
    """Test ModelService behavior when the model file is missing.
    The test patches ``os.path.join`` to redirect the final model path to a
    temporary non‑existent file. The original ``os.path.join`` is saved before
    patching to avoid recursion, and the ModelService singleton is reset so
    that a fresh instance loads the (mocked) path.
    """
    # Ensure a fresh singleton for the test
    from backend.model_service import ModelService
    ModelService._instance = None

    # Force the model loading to point to a non‑existent file
    fake_path = tmp_path / 'nonexistent.pkl'
    # Save the original join function before monkey‑patching
    original_join = os.path.join
    def fake_join(*args, **kwargs):
        # If the final component is 'final_model.pkl', return the fake path
        if args and args[-1] == 'final_model.pkl':
            return str(fake_path)
        # Otherwise delegate to the original implementation
        return original_join(*args, **kwargs)
    # Apply the patch; pytest will automatically restore it after the test
    monkeypatch.setattr(os.path, 'join', fake_join)

    # Re‑instantiate the service – it will attempt to load the fake path
    service = ModelService()
    health = service.health_check()
    assert health['model_loaded'] is False, (
        "Health check should report model not loaded when file missing"
    )
