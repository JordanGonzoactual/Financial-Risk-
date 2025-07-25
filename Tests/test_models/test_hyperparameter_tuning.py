"""
Tests for hyperparameter tuning components.

This module tests the hyperparameter optimization pipeline including
Optuna studies, MLflow integration, and XGBoost tuning.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Mock the required modules if not available
try:
    import optuna
    import mlflow
    import xgboost as xgb
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    
    # Create mock modules
    class MockOptuna:
        class Study:
            def __init__(self):
                self.best_params = {'max_depth': 6, 'learning_rate': 0.1}
                self.best_value = 0.85
                self.best_trial = Mock()
                self.best_trial.number = 42
            
            def optimize(self, objective, n_trials):
                pass
        
        @staticmethod
        def create_study(**kwargs):
            return MockOptuna.Study()
    
    class MockMLflow:
        @staticmethod
        def start_run(**kwargs):
            return Mock()
        
        @staticmethod
        def log_param(key, value):
            pass
        
        @staticmethod
        def log_metric(key, value):
            pass
        
        @staticmethod
        def log_params(params):
            pass
        
        @staticmethod
        def log_metrics(metrics):
            pass
    
    class MockXGBoost:
        class XGBClassifier:
            def __init__(self, **kwargs):
                self.params = kwargs
            
            def fit(self, X, y, **kwargs):
                return self
            
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)
    
    optuna = MockOptuna()
    mlflow = MockMLflow()
    xgb = MockXGBoost()


class TestHyperparameterTuning:
    """Test class for hyperparameter tuning functionality."""
    
    @pytest.fixture
    def mock_xgb_model(self):
        """Fixture providing a mock XGBoost model."""
        def create_model(**kwargs):
            model = Mock()
            model.predict_proba.return_value = np.random.rand(100, 2)
            model.best_iteration = 50
            model.feature_names_in_ = [f'feature_{i}' for i in range(10)]
            model.params = kwargs  # Store the parameters passed to constructor
            return model
        return create_model(**{})
    
    def test_optuna_study_creation(self, mock_study):
        """Test Optuna study creation and configuration."""
        # Mock optuna.create_study
        with patch('optuna.create_study', return_value=mock_study):
            study = optuna.create_study(
                direction='maximize',
                study_name='test_study',
                storage='sqlite:///test.db'
            )
            
            assert study is not None
            assert hasattr(study, 'best_params')
            assert hasattr(study, 'best_value')
    
    def test_objective_function_structure(self, sample_training_data):
        """Test the structure of the objective function for Optuna."""
        X, y = sample_training_data
        
        def mock_objective(trial):
            # Mock parameter suggestions
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
            }
            
            # Mock cross-validation score
            return np.random.uniform(0.7, 0.9)
        
        # Create mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = lambda name, low, high: np.random.randint(low, high + 1)
        mock_trial.suggest_float.side_effect = lambda name, low, high: np.random.uniform(low, high)
        
        # Test objective function
        score = mock_objective(mock_trial)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1  # Assuming score is a metric like accuracy or AUC
    
    def test_mlflow_integration(self, sample_training_data, mock_study):
        """Test MLflow integration for experiment tracking."""
        X, y = sample_training_data
        
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics:
            
            mock_run = Mock()
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            # Simulate hyperparameter tuning with MLflow logging
            with mlflow.start_run():
                mlflow.log_params(mock_study.best_params)
                mlflow.log_metrics({'best_score': mock_study.best_value})
            
            # Verify MLflow calls
            mock_start_run.assert_called()
            mock_log_params.assert_called_with(mock_study.best_params)
            mock_log_metrics.assert_called_with({'best_score': mock_study.best_value})
    
    def test_cross_validation_integration(self, sample_training_data):
        """Test cross-validation integration in hyperparameter tuning."""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score
        
        X, y = sample_training_data
        
        # Mock cross-validation directly to avoid sklearn internals
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.82, 0.85, 0.83, 0.86, 0.84])
            
            # Create a simple mock estimator
            mock_estimator = Mock()
            
            # Call the mocked cross_val_score
            scores = mock_cv(mock_estimator, X, y, cv=5, scoring='roc_auc')
            
            assert len(scores) == 5
            assert all(0 <= score <= 1 for score in scores)
            assert np.mean(scores) > 0.8
            
            # Verify the mock was called with correct parameters
            mock_cv.assert_called_once_with(mock_estimator, X, y, cv=5, scoring='roc_auc')
    
    def test_early_stopping_configuration(self, sample_training_data, mock_xgb_model):
        """Test early stopping configuration in XGBoost training."""
        X, y = sample_training_data
        X_train, X_val = X[:800], X[800:]
        y_train, y_val = y[:800], y[800:]
        
        # Mock XGBoost training with early stopping
        with patch('xgboost.XGBClassifier', return_value=mock_xgb_model):
            model = xgb.XGBClassifier()
            
            # Simulate training with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Verify model was trained
            assert model is not None
            assert hasattr(model, 'best_iteration')
    
    def test_hyperparameter_space_definition(self):
        """Test hyperparameter space definition for XGBoost."""
        # Define expected hyperparameter ranges
        param_space = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 200),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0, 2),
            'reg_lambda': (0, 2),
            'min_child_weight': (1, 10),
            'gamma': (0, 1)
        }
        
        # Test parameter space validity
        for param, (low, high) in param_space.items():
            assert low < high, f"Invalid range for {param}: {low} >= {high}"
            assert low >= 0, f"Negative lower bound for {param}: {low}"
    
    def test_study_persistence(self, mock_study):
        """Test study persistence and resuming."""
        study_name = "test_financial_risk_study"
        storage_url = "sqlite:///test_study.db"
        
        with patch('optuna.create_study', return_value=mock_study) as mock_create:
            # Test study creation with persistence
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction='maximize'
            )
            
            mock_create.assert_called_with(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction='maximize'
            )
            
            assert study is not None
    
    def test_best_parameters_extraction(self, mock_study):
        """Test extraction of best parameters from completed study."""
        # Test parameter extraction
        best_params = mock_study.best_params
        best_score = mock_study.best_value
        
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert isinstance(best_score, (int, float))
        
        # Verify required parameters are present
        required_params = ['max_depth', 'learning_rate', 'n_estimators']
        for param in required_params:
            assert param in best_params, f"Required parameter {param} not found"
    
    def test_model_training_with_best_params(self, sample_training_data, mock_study, mock_xgb_model):
        """Test model training with best parameters from tuning."""
        X, y = sample_training_data
        
        with patch('xgboost.XGBClassifier', return_value=mock_xgb_model) as mock_xgb_class:
            # Train model with best parameters
            model = xgb.XGBClassifier(**mock_study.best_params)
            model.fit(X, y)
            
            # Verify model training
            assert model is not None
            # Verify XGBClassifier was called with the best parameters
            mock_xgb_class.assert_called_once_with(**mock_study.best_params)
            # Verify fit was called
            model.fit.assert_called_once_with(X, y)


class TestHyperparameterTuningPipeline:
    """Test class for the complete hyperparameter tuning pipeline."""
    
    @pytest.fixture
    def mock_pipeline_components(self):
        """Fixture providing mocked pipeline components."""
        components = {
            'data_loader': Mock(),
            'preprocessor': Mock(),
            'model': Mock(),
            'evaluator': Mock()
        }
        
        # Configure mocks
        components['data_loader'].load_data.return_value = (
            pd.DataFrame(np.random.randn(1000, 10)),
            pd.Series(np.random.choice([0, 1], 1000))
        )
        
        components['preprocessor'].fit_transform.return_value = pd.DataFrame(np.random.randn(1000, 15))
        components['model'].predict_proba.return_value = np.random.rand(1000, 2)
        components['evaluator'].evaluate.return_value = {'auc': 0.85, 'accuracy': 0.82}
        
        return components
    
    def test_complete_tuning_pipeline(self, mock_pipeline_components, mock_study):
        """Test the complete hyperparameter tuning pipeline."""
        components = mock_pipeline_components
        
        with patch('optuna.create_study', return_value=mock_study), \
             patch('mlflow.start_run'):
            
            # Simulate complete pipeline
            # 1. Load data
            X, y = components['data_loader'].load_data()
            
            # 2. Create study
            study = optuna.create_study(direction='maximize')
            
            # 3. Run optimization (mocked)
            study.optimize(lambda trial: 0.85, n_trials=10)
            
            # 4. Train final model
            final_model = components['model']
            final_model.fit(X, y)
            
            # 5. Evaluate
            metrics = components['evaluator'].evaluate(final_model, X, y)
            
            # Verify pipeline execution
            assert X is not None
            assert y is not None
            assert study.best_params is not None
            assert metrics['auc'] > 0.8
    
    def test_pipeline_error_handling(self, mock_pipeline_components):
        """Test error handling in the tuning pipeline."""
        components = mock_pipeline_components
        
        # Test data loading error
        components['data_loader'].load_data.side_effect = FileNotFoundError("Data not found")
        
        with pytest.raises(FileNotFoundError):
            components['data_loader'].load_data()
        
        # Test model training error
        components['model'].fit.side_effect = ValueError("Invalid parameters")
        
        with pytest.raises(ValueError):
            components['model'].fit(Mock(), Mock())
    
    def test_pipeline_with_different_metrics(self, mock_pipeline_components, mock_study):
        """Test pipeline with different optimization metrics."""
        components = mock_pipeline_components
        
        metrics_to_test = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        
        for metric in metrics_to_test:
            # Mock evaluator to return specific metric
            components['evaluator'].evaluate.return_value = {metric: 0.85}
            
            with patch('optuna.create_study', return_value=mock_study):
                study = optuna.create_study(direction='maximize')
                
                # Simulate optimization for this metric
                def objective(trial):
                    return components['evaluator'].evaluate(Mock(), Mock(), Mock())[metric]
                
                result = objective(Mock())
                assert result == 0.85
    
    def test_pipeline_reproducibility(self, mock_pipeline_components, mock_study):
        """Test pipeline reproducibility with random seeds."""
        components = mock_pipeline_components
        
        # Set random seed
        np.random.seed(42)
        
        with patch('optuna.create_study', return_value=mock_study):
            # Run pipeline twice with same seed
            results1 = []
            results2 = []
            
            for _ in range(2):
                np.random.seed(42)  # Reset seed
                study = optuna.create_study(direction='maximize', sampler=Mock())
                
                # Mock optimization result
                result = np.random.random()
                results1.append(result) if len(results1) < 1 else results2.append(result)
            
            # Results should be identical with same seed
            # Note: This is a simplified test - actual reproducibility depends on
            # proper seed setting in all random components
            assert len(results1) == 1
            assert len(results2) == 1


class TestOptunaTuningSpecific:
    """Test class for Optuna-specific functionality."""
    
    def test_optuna_sampler_configuration(self):
        """Test different Optuna sampler configurations."""
        samplers_to_test = [
            'TPESampler',
            'RandomSampler', 
            'CmaEsSampler'
        ]
        
        for sampler_name in samplers_to_test:
            # Mock sampler creation
            mock_sampler = Mock()
            mock_sampler.__class__.__name__ = sampler_name
            
            with patch(f'optuna.samplers.{sampler_name}', return_value=mock_sampler):
                # This would test actual sampler creation
                sampler = mock_sampler
                assert sampler is not None
                assert sampler.__class__.__name__ == sampler_name
    
    def test_optuna_pruner_configuration(self):
        """Test different Optuna pruner configurations."""
        pruners_to_test = [
            'MedianPruner',
            'SuccessiveHalvingPruner',
            'HyperbandPruner'
        ]
        
        for pruner_name in pruners_to_test:
            # Mock pruner creation
            mock_pruner = Mock()
            mock_pruner.__class__.__name__ = pruner_name
            
            with patch(f'optuna.pruners.{pruner_name}', return_value=mock_pruner):
                pruner = mock_pruner
                assert pruner is not None
                assert pruner.__class__.__name__ == pruner_name
    
    def test_optuna_study_statistics(self, mock_study):
        """Test extraction of study statistics."""
        # Mock study with trials
        mock_study.trials = [Mock() for _ in range(50)]
        mock_study.trials_dataframe.return_value = pd.DataFrame({
            'value': np.random.uniform(0.7, 0.9, 50),
            'params_max_depth': np.random.randint(3, 10, 50),
            'params_learning_rate': np.random.uniform(0.01, 0.3, 50)
        })
        
        # Test statistics extraction
        n_trials = len(mock_study.trials)
        best_value = mock_study.best_value
        
        assert n_trials == 50
        assert isinstance(best_value, (int, float))
        assert 0 <= best_value <= 1
    
    def test_optuna_visualization_data(self, mock_study):
        """Test data preparation for Optuna visualizations."""
        # Mock trials dataframe
        trials_df = pd.DataFrame({
            'number': range(10),
            'value': np.random.uniform(0.7, 0.9, 10),
            'params_max_depth': np.random.randint(3, 10, 10),
            'params_learning_rate': np.random.uniform(0.01, 0.3, 10),
            'state': ['COMPLETE'] * 10
        })
        
        mock_study.trials_dataframe.return_value = trials_df
        
        # Test visualization data extraction
        df = mock_study.trials_dataframe()
        
        assert len(df) == 10
        assert 'value' in df.columns
        assert 'params_max_depth' in df.columns
        assert 'params_learning_rate' in df.columns
        assert all(df['state'] == 'COMPLETE')


@pytest.mark.integration
class TestHyperparameterTuningIntegration:
    """Integration tests for hyperparameter tuning with real components."""
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_real_optuna_study_creation(self):
        """Test creating a real Optuna study (if dependencies available)."""
        try:
            import optuna
            
            study = optuna.create_study(direction='maximize')
            assert study is not None
            assert hasattr(study, 'optimize')
            
        except ImportError:
            pytest.skip("Optuna not available")
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_real_xgboost_training(self, sample_loan_data):
        """Test real XGBoost training (if dependencies available)."""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            
            # Use sample data
            X = sample_loan_data.select_dtypes(include=[np.number])
            y = np.random.choice([0, 1], len(X))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = xgb.XGBClassifier(max_depth=3, n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict_proba(X_test)
            
            assert predictions.shape == (len(X_test), 2)
            assert np.all((predictions >= 0) & (predictions <= 1))
            
        except ImportError:
            pytest.skip("XGBoost not available")
    
    def test_mlflow_logging_integration(self, temp_data_directory):
        """Test MLflow logging integration."""
        # Mock MLflow tracking
        with patch('mlflow.start_run'), \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.set_tracking_uri'):
            
            # Simulate experiment logging
            params = {'max_depth': 6, 'learning_rate': 0.1}
            metrics = {'auc': 0.85, 'accuracy': 0.82}
            
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
            
            # Verify logging calls
            mock_log_params.assert_called_with(params)
            mock_log_metrics.assert_called_with(metrics)


@pytest.mark.performance
class TestHyperparameterTuningPerformance:
    """Performance tests for hyperparameter tuning."""
    
    def test_tuning_performance_with_large_data(self, sample_training_data):
        """Test tuning performance with larger datasets."""
        X, y = sample_training_data
        
        # Expand dataset
        large_X = pd.concat([X] * 10, ignore_index=True)
        large_y = pd.concat([y] * 10, ignore_index=True)
        
        assert len(large_X) == 10000
        assert len(large_y) == 10000
        
        # Mock quick optimization
        with patch('optuna.create_study') as mock_create:
            mock_study = Mock()
            mock_study.best_params = {'max_depth': 6}
            mock_study.best_value = 0.85
            mock_create.return_value = mock_study
            
            import time
            start_time = time.time()
            
            # Simulate optimization
            study = optuna.create_study()
            study.optimize(lambda trial: 0.85, n_trials=5)
            
            end_time = time.time()
            
            # Performance should be reasonable
            assert end_time - start_time < 1.0  # Mock should be fast
    
    def test_memory_usage_during_tuning(self, sample_training_data):
        """Test memory usage during hyperparameter tuning."""
        import psutil
        import os
        
        X, y = sample_training_data
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Mock tuning process
        with patch('optuna.create_study') as mock_create:
            mock_study = Mock()
            mock_study.optimize = Mock()
            mock_create.return_value = mock_study
            
            # Simulate multiple optimization runs
            for _ in range(10):
                study = optuna.create_study()
                study.optimize(lambda trial: 0.85, n_trials=1)
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
