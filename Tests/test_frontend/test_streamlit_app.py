import sys
import os
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import streamlit as st

# Ensure project root and frontend are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

# Imports from the frontend
from frontend.streamlit_app import RiskAssessmentApp
from frontend.app.views import AppViews
from frontend.app.controller import AppController
from frontend.app.state import AppState
from frontend.app.ui_components import (
    DataRequirementsComponent,
    FileUploadComponent,
    DataPreviewComponent,
    ResultsDashboardComponent,
    DownloadManagerComponent,
)
from frontend.app.api_client import RiskAssessmentClient
from frontend.utils.data_validator import RAW_FEATURE_SCHEMA


# -----------------------
# Helpers and Test Fixtures
# -----------------------
class DummyCM:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class DummyColumn:
    def __init__(self, calls):
        self.calls = calls
    def markdown(self, *args, **kwargs):
        self.calls.append(("markdown", args, kwargs))
    def write(self, *args, **kwargs):
        self.calls.append(("write", args, kwargs))
    def metric(self, *args, **kwargs):
        self.calls.append(("metric", args, kwargs))

class UploadedFile:
    def __init__(self, name, data: bytes):
        self.name = name
        self._data = data
    def getvalue(self):
        return self._data


def make_full_schema_df(n=3):
    """Create a small DataFrame with all required columns."""
    cols = list(RAW_FEATURE_SCHEMA)
    data = {}
    for c in cols:
        # crude defaults: numbers get 1, strings get 'x', dates get '2024-01-01'
        if "Date" in c or "date" in c:
            data[c] = ["2024-01-01"] * n
        elif any(k in c for k in ["Rate", "Ratio", "Score", "Amount", "Income", "Tenure", "Liabilities", "Assets", "Payment", "Loan", "Length"]):
            data[c] = [1] * n
        else:
            data[c] = ["x"] * n
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def reset_session_state(monkeypatch):
    # Provide a clean session state for each test
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    yield


@pytest.fixture
def dummy_context_manager():
    return DummyCM()


# -----------------------
# Tests: RiskAssessmentApp core
# -----------------------
class TestRiskAssessmentApp:
    def test_initialize_success(self, monkeypatch):
        # Patch Streamlit and backend launcher
        set_cfg_calls = []
        monkeypatch.setattr(st, "set_page_config", lambda **kw: set_cfg_calls.append(kw))
        monkeypatch.setattr(st, "spinner", lambda *a, **k: DummyCM())

        # Avoid real backend start and AppState side effects
        import frontend.streamlit_app as sa
        monkeypatch.setattr(sa, "start_flask_backend", lambda: True)
        init_called = {"called": False}
        original_init = sa.AppState.initialize
        def tracking_init():
            init_called["called"] = True
            return original_init()
        monkeypatch.setattr(sa.AppState, "initialize", tracking_init)

        app = RiskAssessmentApp(views=SimpleNamespace(), controller=SimpleNamespace())
        assert app.initialized is True
        assert len(set_cfg_calls) == 1
        assert init_called["called"] is True

    def test_initialize_backend_failure(self, monkeypatch):
        err_msgs = []
        monkeypatch.setattr(st, "set_page_config", lambda **kw: None)
        monkeypatch.setattr(st, "spinner", lambda *a, **k: DummyCM())
        monkeypatch.setattr(st, "error", lambda msg: err_msgs.append(msg))

        import frontend.streamlit_app as sa
        monkeypatch.setattr(sa, "start_flask_backend", lambda: "Failed to start")

        app = RiskAssessmentApp(views=SimpleNamespace(), controller=SimpleNamespace())
        assert app.initialized is False
        assert any("Failed to start" in m for m in err_msgs)

    def test_load_css_success(self, monkeypatch, tmp_path):
        css_file = tmp_path / "style.css"
        css_file.write_text("body { background: #fff; }")
        md_calls = []
        monkeypatch.setattr(st, "markdown", lambda *a, **k: md_calls.append((a, k)))

        app = RiskAssessmentApp(views=SimpleNamespace(), controller=SimpleNamespace())
        app._load_css(str(css_file))
        assert md_calls, "st.markdown should be called to inject CSS"

    def test_load_css_missing(self, monkeypatch):
        import frontend.streamlit_app as sa
        warnings = []
        class LoggerStub:
            def warning(self, msg):
                warnings.append(msg)
        monkeypatch.setattr(sa, "logger", LoggerStub())

        app = RiskAssessmentApp(views=SimpleNamespace(), controller=SimpleNamespace())
        app._load_css("nonexistent.css")
        assert any("CSS file not found" in w for w in warnings)

    def test_run_not_initialized(self, monkeypatch):
        import frontend.streamlit_app as sa
        warns = []
        class LoggerStub:
            def warning(self, msg):
                warns.append(msg)
        monkeypatch.setattr(sa, "logger", LoggerStub())

        app = RiskAssessmentApp(views=SimpleNamespace(), controller=SimpleNamespace())
        app.initialized = False
        app.run()
        assert any("not initialized" in w for w in warns)

    def test_run_success_calls_views(self):
        calls = []
        views = SimpleNamespace(
            render_sidebar=lambda: calls.append("sidebar"),
            render_main_content=lambda: calls.append("main")
        )
        app = RiskAssessmentApp(views=views, controller=SimpleNamespace())
        app.initialized = True
        app.run()
        assert calls == ["sidebar", "main"]

    def test_run_exception_structured_error(self, monkeypatch):
        err_msgs = []
        monkeypatch.setattr(st, "error", lambda msg: err_msgs.append(msg))

        class DummyResponse:
            def json(self):
                return {
                    "category": "missing_columns",
                    "details": "Missing columns detected",
                    "missing_columns": ["Age", "AnnualIncome"],
                }
        class DummyExc(Exception):
            def __init__(self):
                self.response = DummyResponse()
        views = SimpleNamespace(
            render_sidebar=lambda: (_ for _ in ()).throw(DummyExc()),
            render_main_content=lambda: None,
        )
        app = RiskAssessmentApp(views=views, controller=SimpleNamespace())
        app.initialized = True
        app.run()
        assert any("Data Validation Error" in m for m in err_msgs)

    def test_run_exception_unstructured_error(self, monkeypatch):
        err_msgs = []
        monkeypatch.setattr(st, "error", lambda msg: err_msgs.append(msg))

        class DummyResponse:
            def json(self):
                raise ValueError("not json")
        class DummyExc(Exception):
            def __init__(self):
                self.response = DummyResponse()
        views = SimpleNamespace(
            render_sidebar=lambda: (_ for _ in ()).throw(DummyExc()),
            render_main_content=lambda: None,
        )
        app = RiskAssessmentApp(views=views, controller=SimpleNamespace())
        app.initialized = True
        app.run()
        assert any("unexpected error" in m.lower() for m in err_msgs)


# -----------------------
# Tests: UI Components
# -----------------------
class TestUIComponents:
    def test_data_requirements_component_renders(self, monkeypatch):
        calls = []
        monkeypatch.setattr(st, "header", lambda *a, **k: calls.append(("header", a)))
        monkeypatch.setattr(st, "info", lambda *a, **k: calls.append(("info", a)))
        monkeypatch.setattr(st, "success", lambda *a, **k: calls.append(("success", a)))
        monkeypatch.setattr(st, "subheader", lambda *a, **k: calls.append(("subheader", a)))
        # columns grid
        col_calls = []
        monkeypatch.setattr(st, "columns", lambda n: (DummyColumn(col_calls), DummyColumn(col_calls), DummyColumn(col_calls)))
        monkeypatch.setattr(st, "expander", lambda *a, **k: DummyCM())
        monkeypatch.setattr(st, "divider", lambda *a, **k: calls.append(("divider", a)))

        comp = DataRequirementsComponent()
        comp.render()
        assert any(t[0] == "header" for t in calls)
        assert any(t[0] == "success" for t in calls)
        assert len(col_calls) > 0

    def test_data_requirements_component_missing_schema(self, monkeypatch):
        import frontend.app.ui_components as ui
        errs, warns = [], []
        monkeypatch.setattr(st, "error", lambda msg: errs.append(msg))
        monkeypatch.setattr(st, "warning", lambda msg: warns.append(msg))
        monkeypatch.setattr(ui, "RAW_FEATURE_SCHEMA", None)

        comp = DataRequirementsComponent()
        comp.render()
        assert errs and warns

    def test_file_upload_component_csv(self, monkeypatch):
        # Prepare CSV
        df = make_full_schema_df(2)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        uploaded = UploadedFile("sample.csv", csv_bytes)

        # Patch st and state
        frames = []
        msgs = []
        monkeypatch.setattr(st, "file_uploader", lambda *a, **k: uploaded)
        monkeypatch.setattr(st, "dataframe", lambda x, **k: frames.append(x))
        monkeypatch.setattr(st, "success", lambda *a, **k: msgs.append("success"))

        controller = AppController()
        comp = FileUploadComponent(controller)
        comp.render()

        assert AppState.get_state("raw_data") is not None
        assert AppState.get_state("csv_content") is not None
        assert frames, "Preview dataframe should be rendered"

    def test_data_preview_component_validation_flow(self, monkeypatch):
        # Seed state with raw data
        df = make_full_schema_df(1)
        AppState.set_state("raw_data", df)

        # Patch st.button to trigger validation
        btn_calls = []
        monkeypatch.setattr(st, "button", lambda *a, **k: (btn_calls.append("validate"), True)[1])
        monkeypatch.setattr(st, "dataframe", lambda *a, **k: None)
        monkeypatch.setattr(st, "expander", lambda *a, **k: DummyCM())
        monkeypatch.setattr(st, "success", lambda *a, **k: None)
        monkeypatch.setattr(st, "warning", lambda *a, **k: None)
        monkeypatch.setattr(st, "error", lambda *a, **k: None)
        monkeypatch.setattr(st, "rerun", lambda: None, raising=False)

        # Patch controller to set a validation report
        controller = AppController()
        def fake_validate():
            AppState.set_state("validation_report", {"is_valid": True, "errors": [], "warnings": []})
            AppState.set_state("validated_data", df.copy())
        monkeypatch.setattr(controller, "handle_data_validation", fake_validate)

        comp = DataPreviewComponent(controller)
        comp.render()
        assert AppState.get_state("validation_report")["is_valid"] is True
        assert AppState.get_state("validated_data") is not None

    def test_processing_controls_runs_assessment(self, monkeypatch):
        AppState.set_state("validated_data", make_full_schema_df(2))
        AppState.set_state("csv_content", make_full_schema_df(2).to_csv(index=False))

        controller = AppController()
        results_df = pd.DataFrame({"risk_score": [0.1, 0.9]})
        monkeypatch.setattr(controller.api_client, "predict_batch", lambda csv: results_df)

        # Trigger button click
        monkeypatch.setattr(st, "button", lambda *a, **k: True)
        comp = SimpleNamespace(render=lambda: AppController.handle_assessment_processing(controller))
        # Use component API
        from frontend.app.ui_components import ProcessingControlsComponent
        pcomp = ProcessingControlsComponent(controller)
        pcomp.render()
        assert AppState.get_state("assessment_results") is not None

    def test_results_dashboard_component_renders(self, monkeypatch):
        df = pd.DataFrame({
            "risk_score": [0.1, 0.3, 0.7],
            "risk_category": ["High", "Low", "High"],
            "loan_amount": [1000, 2000, 1500],
        })
        AppState.set_state("assessment_results", df)

        tabs_calls, df_calls, metric_calls = [], [], []
        monkeypatch.setattr(st, "tabs", lambda names: (DummyCM(), DummyCM(), DummyCM()))
        monkeypatch.setattr(st, "dataframe", lambda x, **k: df_calls.append(x))
        cols_rec = []
        monkeypatch.setattr(st, "columns", lambda n: (DummyColumn(cols_rec), DummyColumn(cols_rec), DummyColumn(cols_rec)))
        monkeypatch.setattr(st, "plotly_chart", lambda *a, **k: tabs_calls.append("chart"))

        comp = ResultsDashboardComponent()
        comp.render()
        assert df_calls, "Detailed results table should render"
        assert any(call[0] == "metric" for call in cols_rec), "Summary metrics should render"

    def test_download_manager_component(self, monkeypatch):
        df = pd.DataFrame({"a": [1, 2]})
        AppState.set_state("assessment_results", df)
        dl_calls = []
        monkeypatch.setattr(st, "download_button", lambda **kw: dl_calls.append(kw))

        comp = DownloadManagerComponent()
        comp.render()
        assert dl_calls and dl_calls[0]["label"] == "Download as CSV"


# -----------------------
# Tests: Views and Controller
# -----------------------
class TestViewsAndController:
    def test_views_render_sidebar(self, monkeypatch):
        # Sidebar context
        monkeypatch.setattr(st, "sidebar", DummyCM())
        headers, infos = [], []
        monkeypatch.setattr(st, "header", lambda *a, **k: headers.append(a))
        monkeypatch.setattr(st, "info", lambda *a, **k: infos.append(a))

        views = AppViews()
        views.render_sidebar()
        assert headers and infos

    def test_views_render_main_content_order(self, monkeypatch):
        calls = []
        views = AppViews()
        # Replace components with dummies to record order
        views.data_requirements.render = lambda: calls.append("requirements")
        views.file_uploader.render = lambda: calls.append("uploader")
        views.data_preview.render = lambda: calls.append("preview")
        views.results_dashboard.render = lambda: calls.append("results")
        # No results so download not rendered
        AppState.set_state("assessment_results", None)
        views.render_main_content()
        assert calls == ["requirements", "uploader", "preview", "results"]

    def test_controller_handle_file_upload_csv(self):
        controller = AppController()
        df = make_full_schema_df(2)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        uploaded = UploadedFile("data.csv", csv_bytes)

        controller.handle_file_upload(uploaded)
        assert isinstance(AppState.get_state("raw_data"), pd.DataFrame)
        assert isinstance(AppState.get_state("csv_content"), str)

    def test_controller_handle_data_validation_valid(self, monkeypatch):
        controller = AppController()
        AppState.set_state("raw_data", make_full_schema_df(1))

        # Stub CSVValidator within controller module path
        import frontend.app.controller as ctrl_mod
        class DummyVal:
            def __init__(self, df): pass
            def validate(self): return {"is_valid": True, "errors": [], "warnings": []}
            def get_clean_data(self): return make_full_schema_df(1)
        monkeypatch.setattr(ctrl_mod, "CSVValidator", DummyVal)

        controller.handle_data_validation()
        assert AppState.get_state("validated_data") is not None
        assert AppState.get_state("validation_report")["is_valid"] is True

    def test_controller_handle_data_validation_invalid(self, monkeypatch):
        controller = AppController()
        AppState.set_state("raw_data", make_full_schema_df(1))

        import frontend.app.controller as ctrl_mod
        class DummyVal:
            def __init__(self, df): pass
            def validate(self): return {"is_valid": False, "errors": ["bad"], "warnings": []}
            def get_clean_data(self): return None
        monkeypatch.setattr(ctrl_mod, "CSVValidator", DummyVal)

        controller.handle_data_validation()
        assert AppState.get_state("validated_data") is None
        assert AppState.get_state("validation_report")["is_valid"] is False

    def test_controller_handle_assessment_processing_missing_validated(self, monkeypatch):
        controller = AppController()
        AppState.set_state("validated_data", None)
        warns = []
        monkeypatch.setattr(st, "warning", lambda msg: warns.append(msg))
        controller.handle_assessment_processing()
        assert warns and AppState.get_state("assessment_results") is None

    def test_controller_handle_assessment_processing_missing_csv(self, monkeypatch):
        controller = AppController()
        AppState.set_state("validated_data", make_full_schema_df(1))
        AppState.set_state("csv_content", None)
        errs = []
        monkeypatch.setattr(st, "error", lambda msg: errs.append(msg))
        controller.handle_assessment_processing()
        assert errs and AppState.get_state("assessment_results") is None

    def test_controller_handle_assessment_processing_success(self, monkeypatch):
        controller = AppController()
        AppState.set_state("validated_data", make_full_schema_df(1))
        csv_content = make_full_schema_df(1).to_csv(index=False)
        AppState.set_state("csv_content", csv_content)
        succ = []
        monkeypatch.setattr(controller.api_client, "predict_batch", lambda csv: pd.DataFrame({"prediction": [1]}))
        monkeypatch.setattr(st, "success", lambda *a, **k: succ.append("ok"))
        controller.handle_assessment_processing()
        assert AppState.get_state("assessment_results") is not None
        assert succ


# -----------------------
# Tests: AppState
# -----------------------
class TestAppState:
    def test_initialize_get_set_reset(self):
        # Before initialize
        assert st.session_state == {}
        AppState.initialize()
        # Defaults applied
        assert set(AppState._defaults.keys()).issubset(st.session_state.keys())
        # get/set
        AppState.set_state("raw_data", "X")
        assert AppState.get_state("raw_data") == "X"
        # reset
        AppState.reset_all()
        assert AppState.get_state("raw_data") is None


# -----------------------
# Tests: API Client behavior (edge cases)
# -----------------------
class TestApiClient:
    def test_predict_batch_timeout(self, monkeypatch):
        import requests
        client = RiskAssessmentClient(base_url="http://localhost:5001")
        errs = []
        monkeypatch.setattr(st, "error", lambda msg: errs.append(msg))
        class DummyResp:
            def raise_for_status(self):
                raise requests.exceptions.Timeout()
        def fake_post(*a, **k):
            raise requests.exceptions.Timeout()
        monkeypatch.setattr(requests, "post", fake_post)
        res = client.predict_batch("a,b\n1,2\n")
        assert res is None and errs


# -----------------------
# Integration-like smoke test
# -----------------------
class TestIntegrationFlow:
    def test_end_to_end_controller_flow(self, monkeypatch):
        controller = AppController()
        # Simulate upload
        df = make_full_schema_df(2)
        csv_content = df.to_csv(index=False)
        AppState.set_state("raw_data", df)
        AppState.set_state("csv_content", csv_content)
        # Validate
        import frontend.app.controller as ctrl_mod
        class DummyVal:
            def __init__(self, df): pass
            def validate(self): return {"is_valid": True, "errors": [], "warnings": []}
            def get_clean_data(self): return df
        monkeypatch.setattr(ctrl_mod, "CSVValidator", DummyVal)
        controller.handle_data_validation()
        # Process
        monkeypatch.setattr(controller.api_client, "predict_batch", lambda csv: pd.DataFrame({"prediction": [0,1]}))
        controller.handle_assessment_processing()
        results = AppState.get_state("assessment_results")
        assert isinstance(results, pd.DataFrame) and "prediction" in results.columns
