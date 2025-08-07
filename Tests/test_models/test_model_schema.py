import json
import pytest
from pathlib import Path
from backend.model_service import ModelService
from shared.models.raw_data import RawData
from pydantic import ValidationError
from .utils import get_valid_record

# Load the input schema (assumed to be a JSON Schema file)
SCHEMA_PATH = Path(__file__).parents[2] / 'Models' / 'trained_models' / 'input_schema.json'

def load_schema():
    assert SCHEMA_PATH.exists(), f"Schema file not found at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, 'r') as f:
        return json.load(f)

@pytest.fixture(scope='module')
def service():
    return ModelService()

@pytest.fixture(scope='module')
def valid_record():
    """Return a baseline valid record using shared utility."""
    return get_valid_record()

def test_schema_loaded():
    schema = load_schema()
    assert isinstance(schema, dict), "Schema should be a JSON object"
    # Accept either JSON Schema ('required') or legacy format ('required_features')
    assert ('required' in schema) or ('required_features' in schema), (
        "Schema missing required keys: expected 'required' or 'required_features'"
    )

def test_missing_required_field(valid_record):
    # Remove a required field (e.g., CreditScore) and expect validation error
    record = valid_record.copy()
    schema = load_schema()
    required_fields = schema.get('required') or schema.get('required_features', [])
    if 'CreditScore' not in required_fields:
        pytest.skip('CreditScore not listed as required in schema')
    record.pop('CreditScore')
    with pytest.raises(ValidationError):
        RawData(**record)

def test_extra_fields_ignored(valid_record, service):
    # Add an extra field that is not defined in the schema
    record = valid_record.copy()
    record['ExtraFeature'] = 123
    # RawData should raise error if extra fields are not allowed; we test that ModelService can still predict
    # by converting to DataFrame directly (pipeline may drop unknown columns)
    import pandas as pd
    df = pd.DataFrame([record])
    # The service's preprocessing pipeline should handle extra columns gracefully
    preds = service.predict_batch(df)
    assert len(preds) == 1
    assert 0.0 <= preds.iloc[0] <= 100.0

def test_incorrect_data_type(valid_record):
    record = valid_record.copy()
    record['Age'] = 'thirty'  # should be int
    with pytest.raises(ValidationError):
        RawData(**record)

def test_preprocessing_consistency(valid_record, service):
    # Ensure that after preprocessing, the number of features matches the model's expectation
    import pandas as pd
    df = pd.DataFrame([valid_record])
    # Access the internal pipeline directly if available
    pipeline = service.pipeline
    transformed = pipeline.transform(df)
    # The model expects the same number of columns as the training data
    model_input_shape = service.model.get_booster().num_features()
    assert transformed.shape[1] == model_input_shape, (
        f"Preprocessed feature count {transformed.shape[1]} does not match model expectation {model_input_shape}"
    )
