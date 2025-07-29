"""
Comprehensive test suite for the Desktop Application Wrapper.

This module contains tests for the DesktopRiskApp class, covering initialization,
port finding, server startup, window creation, and cleanup functionality.
"""

import os
import sys
import pytest
import socket
import subprocess
import threading
import time
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'frontend'))

from frontend.desktop_app import DesktopRiskApp


@pytest.mark.desktop
@pytest.mark.unit
class TestDesktopRiskAppInitialization:
    """Test class for DesktopRiskApp initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        app = DesktopRiskApp()
        
        assert app.width == 1400
        assert app.height == 900
        assert app.resizable is True
        assert app.streamlit_process is None
        assert app.streamlit_port is None
        assert app.streamlit_url is None
        assert isinstance(app.shutdown_event, threading.Event)
        assert app.frontend_dir == Path(__file__).parent.parent.parent / 'frontend'
        assert app.streamlit_app_path.name == 'streamlit_app.py'
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        app = DesktopRiskApp(width=1200, height=800, resizable=False)
        
        assert app.width == 1200
        assert app.height == 800
        assert app.resizable is False
    
    @patch('signal.signal')
    def test_signal_handler_setup(self, mock_signal):
        """Test that signal handlers are properly set up."""
        app = DesktopRiskApp()
        
        # Verify signal handlers were registered
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list
        assert any(call[0][0] == 2 for call in calls)  # SIGINT
        assert any(call[0][0] == 15 for call in calls)  # SIGTERM
    
    @patch('frontend.desktop_app.DesktopRiskApp._check_python_exe')
    def test_get_python_executable_current_works(self, mock_check_exe):
        """Test Python executable detection when current Python works."""
        app = DesktopRiskApp()
        mock_check_exe.return_value = True

        python_exe = app._get_python_executable()

        assert python_exe == sys.executable
        mock_check_exe.assert_called_once_with(sys.executable)
    
    @patch('pathlib.Path.exists')
    @patch('frontend.desktop_app.DesktopRiskApp._check_python_exe')
    def test_get_python_executable_conda_fallback(self, mock_check_exe, mock_exists):
        """Test Python executable detection falls back to conda environment."""
        app = DesktopRiskApp()

        # Mock current Python fails, conda Python succeeds
        mock_check_exe.side_effect = [False, True, True, True] # sys.executable fails, conda path succeeds
        mock_exists.return_value = True

        python_exe = app._get_python_executable()

        assert "financialRisk" in python_exe
        # Verify that the correct conda path was checked, among others
        assert any("financialRisk" in call.args[0] for call in mock_check_exe.call_args_list)

    @patch('pathlib.Path.exists')
    @patch('frontend.desktop_app.DesktopRiskApp._check_python_exe')
    def test_get_python_executable_not_found(self, mock_check_exe, mock_exists):
        """Test that RuntimeError is raised when no valid Python is found."""
        app = DesktopRiskApp()
        mock_check_exe.return_value = False
        mock_exists.return_value = False # Assume paths do not exist

        with pytest.raises(RuntimeError, match="Failed to find a suitable Python environment"):
            app._get_python_executable()

        # sys.executable is the only one checked as path.exists() is False
        assert mock_check_exe.call_count == 1


@pytest.mark.desktop
@pytest.mark.network
@pytest.mark.unit
class TestPortFinding:
    """Test class for port finding functionality."""
    
    def test_find_available_port_success(self):
        """Test finding an available port successfully."""
        app = DesktopRiskApp()
        
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock
            mock_sock.bind.return_value = None
            
            port = app.find_available_port(start_port=8501)
            
            assert port == 8501
            mock_sock.bind.assert_called_once_with(('localhost', 8501))
    
    def test_find_available_port_retry(self):
        """Test port finding with retries when ports are occupied."""
        app = DesktopRiskApp()
        
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock
            
            # First two ports fail, third succeeds
            mock_sock.bind.side_effect = [OSError(), OSError(), None]
            
            port = app.find_available_port(start_port=8501)
            
            assert port == 8503
            assert mock_sock.bind.call_count == 3
    
    def test_find_available_port_exhausted(self):
        """Test port finding when all ports are exhausted."""
        app = DesktopRiskApp()
        
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock
            mock_sock.bind.side_effect = OSError()
            
            with pytest.raises(RuntimeError, match="No available port found"):
                app.find_available_port(start_port=8501, max_attempts=3)


@pytest.mark.desktop
@pytest.mark.process
@pytest.mark.mock_heavy
@pytest.mark.unit
class TestPythonExecutableChecker:
    """Tests for the _check_python_exe helper method."""

    def test_check_python_exe_success(self):
        """Test the internal checker with a successful mock run."""
        app = DesktopRiskApp()
        
        # We need to test the checker itself, so we mock subprocess.run
        mock_run = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK\n"
        mock_run.return_value = mock_result
        
        result = app._check_python_exe("path/to/python", subprocess_run_func=mock_run)
        
        assert result is True
        mock_run.assert_called_once()

    def test_check_python_exe_failure(self):
        """Test the internal checker with a failed mock run."""
        app = DesktopRiskApp()
        
        mock_run = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        
        result = app._check_python_exe("path/to/python", subprocess_run_func=mock_run)
        
        assert result is False
        mock_run.assert_called_once()




class TestStreamlitServerStartup:
    """Test class for Streamlit server startup functionality."""
    
    @patch('frontend.desktop_app.DesktopRiskApp._get_python_executable')
    @patch('frontend.desktop_app.DesktopRiskApp._wait_for_server_start')
    @patch('subprocess.Popen')
    @patch('frontend.desktop_app.DesktopRiskApp.find_available_port')
    def test_start_streamlit_server_success(self, mock_find_port, mock_popen, mock_wait, mock_get_exe):
        """Test successful Streamlit server startup."""
        app = DesktopRiskApp()
        
        # Setup mocks
        mock_find_port.return_value = 8501
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_wait.return_value = True
        mock_get_exe.return_value = sys.executable

        result = app.start_streamlit_server()
        
        assert result is True
        assert app.streamlit_port == 8501
        assert app.streamlit_url == "http://localhost:8501"
        assert app.streamlit_process == mock_process
        
        # Verify subprocess.Popen was called with correct arguments
        mock_popen.assert_called_once()
        called_cmd = mock_popen.call_args[0][0]
        assert called_cmd[0] == sys.executable
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        
        assert sys.executable in cmd
        assert "-m" in cmd
        assert "streamlit" in cmd
        assert "run" in cmd
        assert "--server.port" in cmd
        assert "8501" in cmd
        assert "--server.headless" in cmd
        assert "true" in cmd
    
    @patch('frontend.desktop_app.DesktopRiskApp.find_available_port')
    def test_start_streamlit_server_port_failure(self, mock_find_port):
        """Test Streamlit server startup when port finding fails."""
        app = DesktopRiskApp()
        
        mock_find_port.side_effect = RuntimeError("No available port")
        
        result = app.start_streamlit_server()
        
        assert result is False
        assert app.streamlit_process is None
    
    @patch('frontend.desktop_app.DesktopRiskApp._wait_for_server_start')
    @patch('subprocess.Popen')
    @patch('frontend.desktop_app.DesktopRiskApp.find_available_port')
    @patch('frontend.desktop_app.DesktopRiskApp._get_python_executable')
    def test_start_streamlit_server_process_failure(self, mock_get_exe, mock_find_port, mock_popen, mock_wait):
        """Test Streamlit server startup when process creation fails."""
        app = DesktopRiskApp()
        
        mock_find_port.return_value = 8501
        mock_popen.side_effect = Exception("Process creation failed")
        mock_get_exe.return_value = sys.executable
        
        result = app.start_streamlit_server()
        
        assert result is False
    
    @patch('requests.get')
    def test_wait_for_server_start_success(self, mock_get):
        """Test successful server startup waiting."""
        app = DesktopRiskApp()
        app.streamlit_url = "http://localhost:8501"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = app._wait_for_server_start(timeout=5)
        
        assert result is True
        mock_get.assert_called_with("http://localhost:8501", timeout=2)
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_wait_for_server_start_timeout(self, mock_sleep, mock_get):
        """Test server startup waiting with timeout."""
        app = DesktopRiskApp()
        app.streamlit_url = "http://localhost:8501"
        
        mock_get.side_effect = Exception("Connection failed")
        
        with patch('time.time', side_effect=[0, 1, 2, 3, 4, 5, 6]):
            result = app._wait_for_server_start(timeout=5)
        
        assert result is False
    
    @patch('requests.get')
    def test_wait_for_server_start_shutdown_event(self, mock_get):
        """Test server startup waiting with shutdown event set."""
        app = DesktopRiskApp()
        app.streamlit_url = "http://localhost:8501"
        app.shutdown_event.set()
        
        result = app._wait_for_server_start(timeout=5)
        
        assert result is False
        mock_get.assert_not_called()


@pytest.mark.desktop
@pytest.mark.gui
@pytest.mark.mock_heavy
@pytest.mark.unit
class TestWebviewWindowCreation:
    """Test class for webview window creation functionality."""
    
    @patch('webview.start')
    @patch('webview.create_window')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_create_window_success(self, mock_shutdown, mock_create_window, mock_start):
        """Test successful window creation."""
        app = DesktopRiskApp(width=1200, height=800, resizable=False)
        app.streamlit_url = "http://localhost:8501"
        
        app.create_window()
        
        mock_create_window.assert_called_once_with(
            title="Financial Risk Assessment",
            url="http://localhost:8501",
            width=1200,
            height=800,
            resizable=False,
            min_size=(800, 600)
        )
        mock_start.assert_called_once_with(debug=False)
        mock_shutdown.assert_called_once()
    
    @patch('webview.create_window')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_create_window_exception(self, mock_shutdown, mock_create_window):
        """Test window creation with exception handling."""
        app = DesktopRiskApp()
        app.streamlit_url = "http://localhost:8501"
        
        mock_create_window.side_effect = Exception("Window creation failed")
        
        app.create_window()
        
        mock_shutdown.assert_called_once()


@pytest.mark.desktop
@pytest.mark.process
@pytest.mark.unit
class TestProcessCleanup:
    """Test class for process cleanup functionality."""
    
    @patch('os.name', 'nt')
    @patch('subprocess.run')
    def test_shutdown_windows(self, mock_run, isolated_desktop_app):
        """Test shutdown behavior on Windows."""
        app = isolated_desktop_app
        mock_process = MagicMock()
        mock_process.pid = 1234
        app.streamlit_process = mock_process

        with patch('os.name', 'nt'):
            app.shutdown()
            
            mock_run.assert_called_once_with([
                'taskkill', '/F', '/T', '/PID', '1234'
            ], check=False)
            
            mock_process.wait.assert_called_once_with(timeout=5)
    
    @patch('os.killpg', create=True)
    @patch('os.getpgid', create=True)
    def test_shutdown_unix(self, mock_getpgid, mock_killpg):
        """Test shutdown process on Unix-like systems."""
        # Create app first, then mock os.name to avoid Path issues
        app = DesktopRiskApp()
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.wait.return_value = None
        app.streamlit_process = mock_process
        mock_getpgid.return_value = 5678
        
        # Mock os.name for the shutdown method only
        with patch('os.name', 'posix'):
            app.shutdown()
        
        assert app.shutdown_event.is_set()
        mock_getpgid.assert_called_once_with(1234)
        mock_killpg.assert_called_once_with(5678, 15)  # SIGTERM = 15
        mock_process.wait.assert_called_once_with(timeout=5)
        assert app.streamlit_process is None
    
    def test_shutdown_timeout_force_kill(self):
        """Test shutdown with timeout leading to force kill."""
        app = DesktopRiskApp()
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.wait.side_effect = subprocess.TimeoutExpired('cmd', 5)
        app.streamlit_process = mock_process
        
        with patch('subprocess.run'):
            app.shutdown()
        
        mock_process.kill.assert_called_once()
        assert app.streamlit_process is None
    
    def test_shutdown_no_process(self):
        """Test shutdown when no process is running."""
        app = DesktopRiskApp()
        app.streamlit_process = None
        
        app.shutdown()
        
        assert app.shutdown_event.is_set()
        # Should not raise any exceptions
    
    def test_signal_handler(self):
        """Test signal handler calls shutdown."""
        app = DesktopRiskApp()
        
        with patch.object(app, 'shutdown') as mock_shutdown:
            app._signal_handler(2, None)  # SIGINT
            mock_shutdown.assert_called_once()


@pytest.mark.desktop
@pytest.mark.integration
@pytest.mark.mock_heavy
class TestDesktopAppRun:
    """Test class for the main run functionality."""
    
    @patch('frontend.desktop_app.DesktopRiskApp.create_window')
    @patch('frontend.desktop_app.DesktopRiskApp.start_streamlit_server')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_run_success(self, mock_shutdown, mock_start_server, mock_create_window, isolated_desktop_app):
        """Test successful application run."""
        app = isolated_desktop_app
        mock_start_server.return_value = True
        
        app.run()
        
        mock_start_server.assert_called_once()
        mock_create_window.assert_called_once()
        mock_shutdown.assert_called_once()
    
    @patch('frontend.desktop_app.DesktopRiskApp.start_streamlit_server')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_run_server_start_failure(self, mock_shutdown, mock_start_server, isolated_desktop_app):
        """Test application run with server startup failure."""
        app = isolated_desktop_app
        mock_start_server.return_value = False
        
        app.run()
        
        mock_start_server.assert_called_once()
        mock_shutdown.assert_called_once()
    
    @patch('frontend.desktop_app.DesktopRiskApp.create_window')
    @patch('frontend.desktop_app.DesktopRiskApp.start_streamlit_server')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_run_keyboard_interrupt(self, mock_shutdown, mock_start_server, mock_create_window, isolated_app_with_process):
        """Test application run with KeyboardInterrupt."""
        isolated_app_with_process.start_streamlit_server = MagicMock(side_effect=KeyboardInterrupt)
        
        isolated_app_with_process.run()
        
        isolated_app_with_process.start_streamlit_server.assert_called_once()
        mock_shutdown.assert_called_once()
    
    @patch('frontend.desktop_app.DesktopRiskApp.create_window')
    @patch('frontend.desktop_app.DesktopRiskApp.start_streamlit_server')
    @patch('frontend.desktop_app.DesktopRiskApp.shutdown')
    def test_run_unexpected_exception(self, mock_shutdown, mock_start_server, mock_create_window, isolated_desktop_app):
        """Test application run with unexpected exception."""
        app = isolated_desktop_app
        mock_start_server.return_value = True
        mock_create_window.side_effect = Exception("Unexpected error")
        
        app.run()
        
        mock_start_server.assert_called_once()
        mock_create_window.assert_called_once()
        mock_shutdown.assert_called_once()


@pytest.mark.desktop
@pytest.mark.unit
class TestMainFunction:
    """Test class for the main function."""
    
    @patch('frontend.desktop_app.DesktopRiskApp')
    def test_main_success(self, mock_app_class):
        """Test successful main function execution."""
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        from frontend.desktop_app import main
        main()
        
        mock_app_class.assert_called_once()
        mock_app.run.assert_called_once()
    
    @patch('frontend.desktop_app.DesktopRiskApp')
    @patch('sys.exit')
    def test_main_exception(self, mock_exit, mock_app_class):
        """Test main function with exception handling."""
        mock_app_class.side_effect = Exception("Initialization failed")
        
        from frontend.desktop_app import main
        main()
        
        mock_exit.assert_called_once_with(1)


@pytest.mark.desktop
@pytest.mark.integration
@pytest.mark.mock_heavy
@pytest.mark.slow
class TestIntegration:
    """Integration tests for the desktop application."""
    
    @patch('webview.start')
    @patch('webview.create_window')
    @patch('requests.get')
    @patch('subprocess.Popen')
    @patch('frontend.desktop_app.DesktopRiskApp.find_available_port')
    def test_full_application_flow(self, mock_find_port, mock_popen, mock_get, 
                                 mock_create_window, mock_start, isolated_desktop_app):
        """Test the complete application flow from start to finish."""
        app = isolated_desktop_app
        
        # Setup mocks for successful flow
        mock_find_port.return_value = 8501
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_popen.return_value = mock_process
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.object(app, 'shutdown') as mock_shutdown:
            app.run()
        
        # Verify the complete flow
        mock_find_port.assert_called_once()
        mock_popen.assert_called_once()
        mock_get.assert_called()
        mock_create_window.assert_called_once()
        mock_start.assert_called_once()
        # Shutdown is called twice: once in create_window finally block, once in run finally block
        assert mock_shutdown.call_count >= 1
        
        assert app.streamlit_port == 8501
        assert app.streamlit_url == "http://localhost:8501"
        assert app.streamlit_process == mock_process


# Fixtures for testing
@pytest.fixture
def desktop_app():
    """Fixture providing a DesktopRiskApp instance for testing."""
    # This fixture is for tests that specifically need to test the initialization process
    return DesktopRiskApp()


@pytest.fixture
def isolated_desktop_app():
    """Fixture providing a DesktopRiskApp instance with Python detection patched out."""
    with patch('frontend.desktop_app.DesktopRiskApp._get_python_executable', return_value=sys.executable) as mock_get_exe:
        app = DesktopRiskApp()
        app.python_executable = mock_get_exe.return_value
        yield app


@pytest.fixture
def mock_streamlit_process():
    """Fixture providing a mock Streamlit process."""
    process = MagicMock()
    process.pid = 1234
    process.wait.return_value = None
    return process


@pytest.fixture
def desktop_app_with_process(desktop_app, mock_streamlit_process):
    """Fixture providing a DesktopRiskApp instance with a mock process."""
    desktop_app.streamlit_process = mock_streamlit_process
    desktop_app.streamlit_port = 8501
    desktop_app.streamlit_url = "http://localhost:8501"
    return desktop_app


@pytest.fixture
def isolated_app_with_process(isolated_desktop_app, mock_streamlit_process):
    """Fixture providing an isolated DesktopRiskApp instance with a mock process."""
    isolated_desktop_app.streamlit_process = mock_streamlit_process
    isolated_desktop_app.streamlit_port = 8501
    isolated_desktop_app.streamlit_url = "http://localhost:8501"
    return isolated_desktop_app
