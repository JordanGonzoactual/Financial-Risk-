# Financial Risk Project - Test Suite

This directory contains a comprehensive test suite for the Financial Risk Assessment project. The tests are organized to mirror the source code structure and provide thorough coverage of all components.

## 📁 Directory Structure

```
Tests/
├── conftest.py                    # Shared pytest fixtures and configuration
├── pytest.ini                    # Pytest configuration file
├── run_tests.py                  # Test runner script
├── README.md                     # This file
├── test_backend/                 # Backend component tests
│   ├── __init__.py
│   ├── test_app.py              # Flask API tests
│   └── test_model_service.py    # Model service tests
├── test_feature_engineering/     # Feature engineering tests
│   ├── __init__.py
│   ├── test_data_loader.py      # Data loading tests
│   └── test_schema_validator.py # Schema validation tests
├── test_models/                  # Model component tests
│   ├── __init__.py
│   └── test_hyperparameter_tuning.py # Hyperparameter tuning tests
└── test_frontend/                # Frontend component tests
    ├── __init__.py
    └── test_streamlit_app.py     # Streamlit app tests
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
cd Tests
python run_tests.py install-deps
```

### 2. Check Test Environment

```bash
python run_tests.py check
```

### 3. Run All Tests

```bash
python run_tests.py all --verbose --coverage
```

## 🧪 Test Categories

### Unit Tests
Test individual functions and methods in isolation.

```bash
python run_tests.py unit --verbose
```

### Integration Tests
Test component interactions and end-to-end workflows.

```bash
python run_tests.py integration --verbose
```

### Performance Tests
Test performance characteristics and resource usage.

```bash
python run_tests.py performance --verbose
```

## 🎯 Component-Specific Testing

### Backend Tests
```bash
python run_tests.py component --component backend --verbose
```

### Feature Engineering Tests
```bash
python run_tests.py component --component feature_engineering --verbose
```

### Model Tests
```bash
python run_tests.py component --component models --verbose
```

### Frontend Tests
```bash
python run_tests.py component --component frontend --verbose
```

## 📊 Test Reports and Coverage

### Generate Comprehensive Report
```bash
python run_tests.py report
```

This generates:
- HTML test report (`Tests/report.html`)
- Coverage HTML report (`Tests/coverage_html/index.html`)
- Coverage XML report (`Tests/coverage.xml`)
- JUnit XML report (`Tests/junit.xml`)

### Run with Coverage Only
```bash
python run_tests.py all --coverage
```

## 🔧 Test Configuration

### Pytest Configuration
The `pytest.ini` file contains:
- Test discovery patterns
- Coverage settings
- Marker definitions
- Output formatting options

### Shared Fixtures
The `conftest.py` file provides:
- Sample loan data fixtures
- Mock model instances
- Temporary directories
- Flask test clients
- Database connections

## 🏷️ Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.model` - Model-related tests
- `@pytest.mark.data` - Data processing tests

### Run Tests by Marker
```bash
# Run only API tests
python -m pytest -m api

# Run only fast tests (exclude slow ones)
python -m pytest -m "not slow"

# Run unit and integration tests
python -m pytest -m "unit or integration"
```

## 🛠️ Advanced Usage

### Parallel Test Execution
```bash
python run_tests.py all --parallel --verbose
```

### Custom Pytest Commands
```bash
# Run specific test file
python -m pytest Tests/test_backend/test_app.py -v

# Run specific test function
python -m pytest Tests/test_backend/test_app.py::TestFlaskApp::test_health_endpoint_success -v

# Run tests matching pattern
python -m pytest -k "test_validation" -v

# Run tests with custom markers
python -m pytest -m "not performance" -v
```

### Debug Mode
```bash
# Run with detailed output and no capture
python -m pytest Tests/ -v -s --tb=long

# Run with Python debugger on failures
python -m pytest Tests/ --pdb
```

## 📋 Test Data and Fixtures

### Sample Data
The test suite includes comprehensive sample data:
- **Loan application data** with all required schema fields
- **Processed feature data** for model testing
- **Mock API responses** for integration testing
- **Invalid data samples** for validation testing

### Mock Objects
Extensive mocking is used for:
- External API calls
- Database connections
- File system operations
- Machine learning models
- Third-party services

## 🔍 Test Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| Backend API | 95%+ | ✅ Comprehensive |
| Feature Engineering | 90%+ | ✅ Comprehensive |
| Model Components | 85%+ | ✅ Comprehensive |
| Frontend | 80%+ | ✅ Comprehensive |
| Integration | 75%+ | ✅ Comprehensive |

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   python run_tests.py install-deps
   ```

3. **Permission Issues**
   ```bash
   # On Windows, run as administrator if needed
   # On Unix, check file permissions
   chmod +x run_tests.py
   ```

4. **Database Connection Issues**
   ```bash
   # Tests use temporary databases by default
   # Check if SQLite is available
   python -c "import sqlite3; print('SQLite available')"
   ```

### Performance Issues

1. **Slow Tests**
   ```bash
   # Run without slow tests
   python -m pytest -m "not slow"
   
   # Show test durations
   python -m pytest --durations=10
   ```

2. **Memory Issues**
   ```bash
   # Run tests in smaller batches
   python -m pytest Tests/test_backend/ -v
   python -m pytest Tests/test_models/ -v
   ```

## 🧹 Maintenance

### Clean Test Artifacts
```bash
python run_tests.py clean
```

This removes:
- `__pycache__` directories
- `.pytest_cache` directories
- Coverage reports
- Test result files

### Update Test Dependencies
```bash
pip install --upgrade pytest pytest-cov pytest-html pytest-xdist
```

## 📚 Best Practices

### Writing Tests

1. **Use descriptive test names**
   ```python
   def test_loan_risk_prediction_with_valid_data(self):
       # Clear what the test does
   ```

2. **Follow AAA pattern**
   ```python
   def test_example(self):
       # Arrange
       data = create_test_data()
       
       # Act
       result = process_data(data)
       
       # Assert
       assert result.is_valid
   ```

3. **Use appropriate fixtures**
   ```python
   def test_with_sample_data(self, sample_loan_data):
       # Use shared fixtures from conftest.py
   ```

4. **Mock external dependencies**
   ```python
   @patch('external_service.api_call')
   def test_with_mocked_service(self, mock_api):
       # Test in isolation
   ```

### Test Organization

1. **Group related tests in classes**
2. **Use meaningful test markers**
3. **Keep tests independent**
4. **Test both success and failure cases**
5. **Include edge cases and boundary conditions**

## 📞 Support

For questions about the test suite:

1. Check this README first
2. Review existing test examples
3. Check pytest documentation
4. Review the project's main documentation

## 🔄 Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    cd Tests
    python run_tests.py all --coverage --parallel
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: Tests/coverage.xml
```

## 📈 Metrics and Reporting

The test suite generates comprehensive metrics:

- **Test execution time**
- **Code coverage percentage**
- **Test success/failure rates**
- **Performance benchmarks**
- **Memory usage statistics**

All metrics are available in the generated reports and can be integrated with monitoring systems.
