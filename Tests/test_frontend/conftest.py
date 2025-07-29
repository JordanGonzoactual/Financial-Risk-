"""
Pytest configuration and fixtures specifically for frontend tests.

This module provides fixtures and configuration for frontend-specific tests,
including desktop application tests with minimal dependencies.
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'frontend'))


@pytest.fixture
def mock_webview():
    """Mock webview module for testing without GUI dependencies."""
    with patch('webview.create_window') as mock_create, \
         patch('webview.start') as mock_start:
        mock_webview_module = MagicMock()
        mock_webview_module.create_window = mock_create
        mock_webview_module.start = mock_start
        yield mock_webview_module


@pytest.fixture
def mock_subprocess():
    """Mock subprocess module for testing without actual process creation."""
    with patch('subprocess.Popen') as mock_popen, \
         patch('subprocess.run') as mock_run:
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process
        
        mock_subprocess_module = MagicMock()
        mock_subprocess_module.Popen = mock_popen
        mock_subprocess_module.run = mock_run
        mock_subprocess_module.TimeoutExpired = subprocess.TimeoutExpired
        mock_subprocess_module.CREATE_NEW_PROCESS_GROUP = 0x00000200
        
        yield mock_subprocess_module


@pytest.fixture
def mock_requests():
    """Mock requests module for testing without network calls."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        mock_requests_module = MagicMock()
        mock_requests_module.get = mock_get
        mock_requests_module.exceptions = requests.exceptions
        
        yield mock_requests_module


@pytest.fixture
def mock_socket():
    """Mock socket module for testing port availability."""
    with patch('socket.socket') as mock_socket_class:
        mock_sock = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_sock
        mock_sock.bind.return_value = None
        
        mock_socket_module = MagicMock()
        mock_socket_module.socket = mock_socket_class
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1
        
        yield mock_socket_module


@pytest.fixture
def desktop_app_instance():
    """Provide a DesktopRiskApp instance for testing."""
    # Import here to avoid import errors in test discovery
    try:
        from frontend.desktop_app import DesktopRiskApp
        return DesktopRiskApp()
    except ImportError:
        # Return a mock if import fails
        return MagicMock()


@pytest.fixture
def mock_streamlit_process():
    """Provide a mock Streamlit process for testing."""
    process = MagicMock()
    process.pid = 1234
    process.wait.return_value = None
    process.poll.return_value = None
    return process


@pytest.fixture
def desktop_app_with_process(desktop_app_instance, mock_streamlit_process):
    """Provide a DesktopRiskApp instance with a running process."""
    desktop_app_instance.streamlit_process = mock_streamlit_process
    desktop_app_instance.streamlit_port = 8501
    desktop_app_instance.streamlit_url = "http://localhost:8501"
    return desktop_app_instance


@pytest.fixture
def mock_logger():
    """Mock logger for testing without actual logging."""
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_desktop_dependencies():
    """Automatically mock desktop dependencies for all tests."""
    with patch('signal.signal'), \
         patch('os.name', 'nt'), \
         patch('sys.executable', 'python'):
        yield


@pytest.fixture
def temp_frontend_dir(tmp_path):
    """Create a temporary frontend directory structure for testing."""
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    
    # Create a mock streamlit_app.py
    streamlit_app = frontend_dir / "streamlit_app.py"
    streamlit_app.write_text("# Mock Streamlit app\nprint('Hello, World!')")
    
    # Create assets directory
    assets_dir = frontend_dir / "assets"
    assets_dir.mkdir()
    
    return frontend_dir


# Import subprocess for exception classes
try:
    import subprocess
    import requests
    import socket
except ImportError:
    # Create mock modules if imports fail
    class MockSubprocess:
        class TimeoutExpired(Exception):
            def __init__(self, cmd, timeout):
                self.cmd = cmd
                self.timeout = timeout
        
        CREATE_NEW_PROCESS_GROUP = 0x00000200
    
    class MockRequests:
        class exceptions:
            class RequestException(Exception):
                pass
    
    subprocess = MockSubprocess()
    requests = MockRequests()
    socket = MagicMock()


# Configure pytest for desktop tests
def pytest_configure(config):
    """Configure pytest for desktop application tests."""
    config.addinivalue_line(
        "markers", "desktop: mark test as desktop application test"
    )
    config.addinivalue_line(
        "markers", "gui: mark test as GUI-related test"
    )
    config.addinivalue_line(
        "markers", "process: mark test as process management test"
    )
    config.addinivalue_line(
        "markers", "network: mark test as network-related test"
    )
    config.addinivalue_line(
        "markers", "mock_heavy: mark test as having extensive mocking"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add desktop marker to all tests in desktop test files
        if "desktop_app" in str(item.fspath):
            item.add_marker(pytest.mark.desktop)
        
        # Add gui marker to window-related tests
        if "window" in item.name.lower() or "webview" in item.name.lower():
            item.add_marker(pytest.mark.gui)
        
        # Add process marker to process-related tests
        if "process" in item.name.lower() or "server" in item.name.lower():
            item.add_marker(pytest.mark.process)
        
        # Add network marker to network-related tests
        if "port" in item.name.lower() or "network" in item.name.lower():
            item.add_marker(pytest.mark.network)
