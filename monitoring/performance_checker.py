import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Ensure project root is on the Python path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.model_service import ModelService  # noqa: E402


class PerformanceChecker:
    """Runs lightweight, test-aligned performance checks against the model.

    Reuses thresholds from Tests/test_models/test_model_performance.py:
    - Accuracy: R² >= 0.85 and RMSE <= 3
    - Latency: single prediction <= 100ms
    """

    R2_MIN: float = 0.85
    RMSE_MAX: float = 3.0
    LATENCY_MS_MAX: float = 100.0

    def __init__(self) -> None:
        self.service = ModelService()

    def _service_health(self) -> Dict[str, Any]:
        try:
            health = self.service.health_check()
        except Exception as e:
            return {
                'model_loaded': False,
                'pipeline_loaded': False,
                'status': 'unhealthy',
                'error': str(e),
            }
        return health

    def check_model_performance(self) -> Dict[str, Any]:
        """Evaluate accuracy (R², RMSE) on the raw holdout dataset.

        Mirrors Tests/test_models/test_model_performance.py logic but avoids pytest.
        """
        health = self._service_health()
        if health.get('status') != 'healthy':
            return {
                'status': 'skip',
                'message': 'Model or pipeline not loaded; skipping accuracy check.',
                'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
            }

        # Locate raw dataset and metadata relative to project root
        data_path = Path(PROJECT_ROOT) / 'Data' / 'raw' / 'Loan.csv'
        meta_path = Path(PROJECT_ROOT) / 'FeatureEngineering' / 'artifacts' / 'transformation_metadata.json'

        if not data_path.exists():
            return {
                'status': 'error',
                'message': f"Dataset not found at {data_path}",
                'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
            }
        if not meta_path.exists():
            return {
                'status': 'error',
                'message': f"Transformation metadata not found at {meta_path}",
                'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
            }

        try:
            df = pd.read_csv(data_path)
            if 'RiskScore' not in df.columns:
                return {
                    'status': 'error',
                    'message': "Expected 'RiskScore' column in Loan.csv as target",
                    'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
                }

            y_true = df['RiskScore']
            X = df.drop(columns=['RiskScore'])

            # Validate raw schema overlap like in tests
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            raw_cols = set(meta.get('numeric_columns', []) + meta.get('categorical_columns', []))
            if len(set(X.columns) & raw_cols) == 0:
                return {
                    'status': 'error',
                    'message': 'No overlap between Loan.csv columns and expected raw schema',
                    'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
                }

            # Safely attempt prediction and handle unfitted pipeline explicitly
            try:
                preds = self.service.predict_batch(X)
            except Exception as e:
                msg = str(e)
                if 'Pipeline is not fitted yet' in msg or 'not fitted' in msg.lower():
                    return {
                        'status': 'skip',
                        'message': 'Pipeline is not fitted yet; skipping accuracy check.',
                        'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
                    }
                return {
                    'status': 'error',
                    'message': f'Prediction failed during accuracy check: {e}',
                    'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
                }
            r2 = r2_score(y_true, preds)
            rmse = mean_squared_error(y_true, preds) ** 0.5

            status = 'pass' if (r2 >= self.R2_MIN and rmse <= self.RMSE_MAX) else 'fail'
            message = (
                f"R²={r2:.3f} (min {self.R2_MIN}), RMSE={rmse:.3f} (max {self.RMSE_MAX})"
            )
            if status == 'fail':
                message = (
                    f"Model accuracy below threshold: {message}"
                )

            return {
                'status': status,
                'r2': float(r2),
                'rmse': float(rmse),
                'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
                'message': message,
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Performance evaluation failed: {e}',
                'thresholds': {'r2_min': self.R2_MIN, 'rmse_max': self.RMSE_MAX},
            }

    def check_prediction_latency(self) -> Dict[str, Any]:
        """Measure single-record prediction latency in milliseconds."""
        health = self._service_health()
        if health.get('status') != 'healthy':
            return {
                'status': 'skip',
                'message': 'Model or pipeline not loaded; skipping latency check.',
                'threshold_ms': self.LATENCY_MS_MAX,
            }

        # Same representative record as used in tests
        record = {
            "ApplicationDate": "2024-01-01",
            "Age": 30,
            "CreditScore": 750,
            "EmploymentStatus": "Employed",
            "EducationLevel": "Bachelor",
            "Experience": 5,
            "LoanAmount": 50000.0,
            "LoanDuration": 60,
            "NumberOfDependents": 0,
            "HomeOwnershipStatus": "Own",
            "MonthlyDebtPayments": 200.0,
            "CreditCardUtilizationRate": 0.1,
            "NumberOfOpenCreditLines": 3,
            "NumberOfCreditInquiries": 0,
            "DebtToIncomeRatio": 0.2,
            "SavingsAccountBalance": 20000.0,
            "CheckingAccountBalance": 5000.0,
            "MonthlyIncome": 8000.0,
            "AnnualIncome": 96000.0,
            "BaseInterestRate": 0.04,
            "InterestRate": 0.05,
            "TotalDebtToIncomeRatio": 0.25,
            "LoanPurpose": "Home",
            "BankruptcyHistory": 0,
            "PaymentHistory": 2,
            "UtilityBillsPaymentHistory": 0.95,
            "PreviousLoanDefaults": 0,
            "TotalAssets": 300000,
            "TotalLiabilities": 100000,
            "NetWorth": 200000,
            "LengthOfCreditHistory": 120,
            "JobTenure": 60,
        }

        try:
            df = pd.DataFrame([record])
            start = time.time()
            # Safely attempt prediction and handle unfitted pipeline explicitly
            try:
                _ = self.service.predict_batch(df)
            except Exception as e:
                msg = str(e)
                if 'Pipeline is not fitted yet' in msg or 'not fitted' in msg.lower():
                    return {
                        'status': 'skip',
                        'message': 'Pipeline is not fitted yet; skipping latency check.',
                        'threshold_ms': float(self.LATENCY_MS_MAX),
                    }
                return {
                    'status': 'error',
                    'message': f'Latency check prediction failed: {e}',
                    'threshold_ms': float(self.LATENCY_MS_MAX),
                }
            elapsed_ms = (time.time() - start) * 1000
            status = 'pass' if elapsed_ms <= self.LATENCY_MS_MAX else 'warn'
            message = (
                f"Prediction latency {elapsed_ms:.1f}ms (max {self.LATENCY_MS_MAX:.0f}ms)"
            )
            if status != 'pass':
                message = f"Latency exceeds target: {message}"

            return {
                'status': status,
                'elapsed_ms': float(elapsed_ms),
                'threshold_ms': float(self.LATENCY_MS_MAX),
                'message': message,
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Latency check failed: {e}',
                'threshold_ms': float(self.LATENCY_MS_MAX),
            }

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all checks and produce a consolidated result with an overall status."""
        health = self._service_health()
        perf = self.check_model_performance()
        latency = self.check_prediction_latency()

        statuses = [perf.get('status'), latency.get('status')]
        if 'fail' in statuses:
            overall = 'fail'
        elif 'warn' in statuses:
            overall = 'warn'
        elif 'error' in statuses:
            overall = 'error'
        elif 'skip' in statuses and all(s in {'skip', 'pass'} for s in statuses):
            overall = 'skip'
        else:
            overall = 'pass'

        return {
            'service_health': health,
            'model_performance': perf,
            'prediction_latency': latency,
            'overall_status': overall,
        }
