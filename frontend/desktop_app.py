"""
Desktop Application Wrapper for Financial Risk Assessment

This module provides a desktop application wrapper that launches the Streamlit
web application in a native desktop window using webview.
"""

import os
import sys
import socket
import subprocess
import threading
import time
import signal
import logging
from typing import Optional, Tuple
from pathlib import Path

# Handle optional dependencies with graceful fallbacks
try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    webview = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Add the frontend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logging import get_logger

logger = get_logger(__name__)


class DesktopRiskApp:
    """
    Desktop application wrapper for the Financial Risk Assessment application.
    
    This class manages the Streamlit server process and creates a native desktop
    window using webview to provide a seamless desktop experience.
    """
    
    def __init__(self, width: int = 1400, height: int = 900, resizable: bool = True):
        """
        Initialize the desktop application.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            resizable: Whether the window should be resizable
        """
        self.width = width
        self.height = height
        self.resizable = resizable
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.streamlit_port: Optional[int] = None
        self.streamlit_url: Optional[str] = None
        self.shutdown_event = threading.Event()
        
        # Get the frontend directory path
        self.frontend_dir = Path(__file__).parent.absolute()
        self.streamlit_app_path = self.frontend_dir / "streamlit_app.py"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Check dependencies on initialization
        self._check_dependencies()
        
        # Detect the correct Python interpreter
        self.python_executable = None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        warnings = []
        
        if not WEBVIEW_AVAILABLE:
            warnings.append("pywebview (will fallback to browser)")
        
        if not REQUESTS_AVAILABLE:
            missing_deps.append("requests")
        
        if warnings:
            warning_msg = f"Optional dependencies missing: {', '.join(warnings)}"
            logger.warning(warning_msg)
        
        if missing_deps:
            error_msg = f"Missing critical dependencies: {', '.join(missing_deps)}\n"
            error_msg += "Please install them using:\n"
            error_msg += f"pip install {' '.join(missing_deps)}"
            logger.error(error_msg)
            raise ImportError(error_msg)
    
    def _check_python_exe(self, python_exe: str, subprocess_run_func=subprocess.run) -> bool:
        """Check if a Python executable has the required packages."""
        try:
            result = subprocess_run_func([
                python_exe, '-c',
                'import pandas, streamlit, requests; print("OK")'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'OK' in result.stdout:
                return True
        except Exception as e:
            logger.warning(f"Python check failed for {python_exe}: {e}")
        return False

    def _get_python_executable(self) -> str:
        """Detect the correct Python executable to use for Streamlit."""
        # Check for container environment override
        if os.environ.get('CONTAINER_ENV'):
            logger.info("Container environment detected. Using base Conda Python.")
            return '/opt/conda/bin/python'

        # First, try to use the same Python that's running this script
        current_python = sys.executable
        if self._check_python_exe(current_python):
            logger.info(f"Using current Python interpreter: {current_python}")
            return current_python

        # Fallback to known conda environment paths
        conda_env_name = "financialRisk"
        home_dir = Path.home()
        
        # Common conda paths
        conda_paths = [
            # Windows
            Path(f"{os.environ.get('CONDA_PREFIX', '')}/../envs/{conda_env_name}/python.exe"),
            home_dir / f"anaconda3/envs/{conda_env_name}/python.exe",
            home_dir / f"miniconda3/envs/{conda_env_name}/python.exe",
            # Unix-like
            Path(f"/opt/anaconda3/envs/{conda_env_name}/bin/python"),
            home_dir / f"anaconda/envs/{conda_env_name}/bin/python"
        ]
        
        for path in conda_paths:
            if path.exists() and self._check_python_exe(str(path)):
                logger.info(f"Found valid Python in conda env: {path}")
                return str(path)
        
        logger.error("Could not find a valid Python executable with required packages.")
        raise RuntimeError("Failed to find a suitable Python environment for Streamlit.")
    
    def find_available_port(self, start_port: int = 8501, max_attempts: int = 100) -> int:
        """
        Find an available port starting from the given port.
        
        Args:
            start_port: Starting port number to check
            max_attempts: Maximum number of ports to check
            
        Returns:
            Available port number
            
        Raises:
            RuntimeError: If no available port is found
        """
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    logger.info(f"Found available port: {port}")
                    return port
            except OSError:
                continue
        
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")
    
    def start_streamlit_server(self) -> bool:
        """
        Start the Streamlit server in a background process.

        Returns:
            True if server started successfully, False otherwise
        """
        try:
            # Find an available port
            self.python_executable = self._get_python_executable()
            self.streamlit_port = self.find_available_port()
            self.streamlit_url = f"http://localhost:{self.streamlit_port}"

            # Prepare the command to start Streamlit
            cmd = [
                self.python_executable, "-m", "streamlit", "run",
                str(self.streamlit_app_path),
                "--server.port", str(self.streamlit_port),
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false",
                "--browser.gatherUsageStats", "false"
            ]

            logger.info(f"Starting Streamlit server with command: {' '.join(cmd)}")

            # Start the Streamlit process
            self.streamlit_process = subprocess.Popen(
                cmd,
                cwd=str(self.frontend_dir),
                # Allow subprocess to inherit stdout/stderr for debugging
                stdout=None,
                stderr=None,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

            # Wait for the server to start
            return self._wait_for_server_start()

        except Exception as e:
            logger.error(f"Failed to start Streamlit server: {e}")
            return False
    
    def _wait_for_server_start(self, timeout: int = 30) -> bool:
        """
        Wait for the Streamlit server to start and become responsive.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is responsive, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
                
            try:
                if not REQUESTS_AVAILABLE:
                    logger.warning("Requests module not available, skipping server health check")
                    time.sleep(2)  # Give server time to start
                    return True
                
                response = requests.get(self.streamlit_url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"Streamlit server is ready at {self.streamlit_url}")
                    return True
            except Exception:
                pass
            
            time.sleep(1)
        
        logger.error(f"Streamlit server failed to start within {timeout} seconds")
        return False
    
    def create_window(self) -> None:
        """
        Create and display the native desktop window.
        """
        try:
            logger.info("Creating desktop window...")
            
            if not WEBVIEW_AVAILABLE:
                logger.error("Webview module not available. Opening browser instead...")
                import webbrowser
                webbrowser.open(self.streamlit_url)
                logger.info(f"Opened {self.streamlit_url} in default browser")
                # Keep the process running
                input("Press Enter to stop the application...")
                return
            
            # Create the webview window
            webview.create_window(
                title="Financial Risk Assessment",
                url=self.streamlit_url,
                width=self.width,
                height=self.height,
                resizable=self.resizable,
                min_size=(800, 600)
            )
            
            # Start the webview (this will block until the window is closed)
            webview.start(debug=False)
            
        except Exception as e:
            logger.error(f"Failed to create desktop window: {e}")
        finally:
            # Ensure cleanup happens when window is closed
            self.shutdown()
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the application and clean up resources.
        """
        logger.info("Shutting down desktop application...")
        self.shutdown_event.set()
        
        # Terminate the Streamlit process
        if self.streamlit_process:
            try:
                if os.name == 'nt':
                    # On Windows, use taskkill to terminate the process tree
                    subprocess.run([
                        'taskkill', '/F', '/T', '/PID', str(self.streamlit_process.pid)
                    ], check=False)
                else:
                    # On Unix-like systems, terminate the process group
                    os.killpg(os.getpgid(self.streamlit_process.pid), signal.SIGTERM)
                
                # Wait for process to terminate
                self.streamlit_process.wait(timeout=5)
                logger.info("Streamlit server terminated successfully")
                
            except subprocess.TimeoutExpired:
                logger.warning("Streamlit server did not terminate gracefully, forcing kill")
                self.streamlit_process.kill()
            except Exception as e:
                logger.error(f"Error terminating Streamlit server: {e}")
            finally:
                self.streamlit_process = None
    
    def run(self) -> None:
        """
        Run the desktop application.

        This method starts the Streamlit server. In a desktop environment, it also
        creates the desktop window. In a container, it runs in web-only mode.
        """
        logger.info("Starting Financial Risk Assessment Application")

        try:
            # Start the Streamlit server
            if not self.start_streamlit_server():
                logger.error("Failed to start Streamlit server")
                return

            # Check for container environment to determine execution mode
            if os.environ.get('CONTAINER_ENV'):
                logger.info("Running in container (web-only mode).")
                logger.info(f"Access the application at {self.streamlit_url}")
                logger.info("Application is running. Press Ctrl+C to stop.")
                # Wait for shutdown signal
                self.shutdown_event.wait()
            else:
                logger.info("Running in desktop mode.")
                # Create and show the desktop window
                self.create_window()

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in desktop application: {e}")
        finally:
            self.shutdown()


def main():
    """Main entry point for the desktop application."""
    try:
        app = DesktopRiskApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start desktop application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
