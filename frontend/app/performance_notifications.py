import streamlit as st
from typing import Optional, Dict, Any

# Ensure runtime import works when running via streamlit_app.py
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from monitoring.performance_checker import PerformanceChecker  # noqa: E402
from .state import AppState  # noqa: E402


class PerformanceNotificationComponent:
    """Handles performance checks and displays user-facing notifications."""

    def __init__(self) -> None:
        self.checker = PerformanceChecker()

    def _render_messages(self, results: Dict[str, Any]) -> None:
        perf = results.get('model_performance', {})
        latency = results.get('prediction_latency', {})
        health = results.get('service_health', {})

        # Critical failures (accuracy below threshold)
        if perf.get('status') == 'fail':
            st.error(
                f"Model performance below threshold. {perf.get('message', '')}\n"
                f"Thresholds: R² ≥ {perf.get('thresholds',{}).get('r2_min')}, "
                f"RMSE ≤ {perf.get('thresholds',{}).get('rmse_max')}"
            )

        # Warnings for latency degradation
        if latency.get('status') in {'warn'}:
            st.warning(latency.get('message', 'Latency degradation detected.'))

        # Errors and skips surfaced as info/warning depending on case
        for k in ('model_performance', 'prediction_latency'):
            item = results.get(k, {})
            if item.get('status') == 'error':
                st.warning(f"{k.replace('_',' ').title()} error: {item.get('message')}")
            if item.get('status') == 'skip':
                st.info(item.get('message'))

        # Success when all pass
        if results.get('overall_status') == 'pass':
            st.success("Model health looks good. All checks passed.")

    def check_and_notify(self) -> Dict[str, Any]:
        """Run lightweight performance checks and display notifications. Returns raw results.

        Quick diagnostic mode:
        - Always runs service health check
        - Runs prediction latency check if service is healthy
        - Skips full accuracy (R²/RMSE) check to avoid heavy operations
        """
        try:
            health = self.checker._service_health()
        except Exception as e:  # Extra safety, though _service_health guards internally
            health = {
                'model_loaded': False,
                'pipeline_loaded': False,
                'status': 'unhealthy',
                'error': str(e),
            }

        user_has_data = (
            AppState.get_state('raw_data') is not None
            or AppState.get_state('assessment_results') is not None
        )

        if health.get('status') != 'healthy':
            latency = {
                'status': 'skip',
                'message': 'Service unhealthy; skipping latency check.',
                'threshold_ms': float(getattr(self.checker, 'LATENCY_MS_MAX', 100.0)),
            }
        elif not user_has_data:
            # Avoid running predictions before any user data has been uploaded/processed
            latency = {
                'status': 'skip',
                'message': 'No user data loaded yet; skipping latency check.',
                'threshold_ms': float(getattr(self.checker, 'LATENCY_MS_MAX', 100.0)),
            }
        else:
            latency = self.checker.check_prediction_latency()

        # Provide a clear skip for accuracy in quick diagnostic mode
        perf = {
            'status': 'skip',
            'message': 'Accuracy check skipped in quick diagnostic mode.',
            'thresholds': {
                'r2_min': float(getattr(self.checker, 'R2_MIN', 0.85)),
                'rmse_max': float(getattr(self.checker, 'RMSE_MAX', 3.0)),
            },
        }

        statuses = [latency.get('status')]
        if health.get('status') != 'healthy':
            statuses.append('error')
        if 'error' in statuses:
            overall = 'error'
        elif 'warn' in statuses:
            overall = 'warn'
        elif 'skip' in statuses and all(s in {'skip', 'pass'} for s in statuses):
            overall = 'skip'
        else:
            overall = 'pass'

        results = {
            'service_health': health,
            'model_performance': perf,
            'prediction_latency': latency,
            'overall_status': overall,
        }

        self._render_messages(results)
        return results
