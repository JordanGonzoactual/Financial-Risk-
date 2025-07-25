import subprocess
import requests
import time
import sys
import os
import atexit
import threading
import queue
import importlib.util

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5001
FLASK_URL = f"http://{FLASK_HOST}:{FLASK_PORT}"
HEALTH_CHECK_URL = f"{FLASK_URL}/health"

_flask_process = None

def diagnose_flask_requirements():
    """Diagnoses if the Flask backend can be started."""
    # 1. Check for required Python packages
    required_packages = ["flask", "pandas", "xgboost"]
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    if missing_packages:
        return f"Missing required Python packages: {', '.join(missing_packages)}"

    # 2. Check for the model file
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        trained_models_dir = os.path.join(project_root, 'Models', 'trained_models')
        if not os.path.exists(trained_models_dir):
            return f"Trained models directory not found: {trained_models_dir}"

        model_file = os.path.join(trained_models_dir, 'final_model.pkl')
        if not os.path.exists(model_file):
            return f"Model file not found: {model_file}"
    except Exception as e:
        return f"An error occurred during model file check: {e}"

    return None

def is_flask_running():
    """Check if the Flask backend is running."""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=1.0)
        if response.status_code == 200:
            print("Flask health check successful.")
            return True
        return False
    except requests.exceptions.RequestException:
        return False

def start_flask_backend():
    """Starts the Flask backend server, captures output, and performs diagnostics."""
    global _flask_process

    if is_flask_running():
        print("Flask backend is already running.")
        return True

    # Perform diagnostic checks before starting
    diagnostic_error = diagnose_flask_requirements()
    if diagnostic_error:
        print(f"Flask prerequisites check failed: {diagnostic_error}")
        return diagnostic_error

    print("Starting Flask backend...")
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
    
    python_executable = sys.executable

    try:
        _flask_process = subprocess.Popen(
            [python_executable, "-m", "flask", "run", f"--port={FLASK_PORT}"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        print(f"Flask backend process started with PID: {_flask_process.pid}")
        atexit.register(kill_flask_backend)
    except FileNotFoundError:
        print(f"Error: Could not find the backend directory at {backend_dir}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while starting Flask: {e}")
        return False

    # Thread to capture and process Flask's output
    output_queue = queue.Queue()
    def enqueue_output(stream, q):
        for line in iter(stream.readline, ''):
            q.put(line)
        stream.close()

    output_thread = threading.Thread(target=enqueue_output, args=(_flask_process.stdout, output_queue))
    output_thread.daemon = True
    output_thread.start()

    # Wait for the backend to become available
    max_wait_time = 30  # seconds
    start_time = time.time()
    wait_interval = 0.2  # initial wait time
    captured_output = []

    while time.time() - start_time < max_wait_time:
        if is_flask_running():
            print("Flask backend started successfully.")
            return True

        # Check for any output from Flask during the wait
        while not output_queue.empty():
            line = output_queue.get_nowait().strip()
            if line:
                print(f"[Flask Output] {line}")
                captured_output.append(line)

        print(f"Waiting for Flask backend... retrying in {wait_interval:.2f} seconds.")
        time.sleep(wait_interval)
        wait_interval = min(wait_interval * 1.5, 5)

    # If the loop finishes, the backend failed to start
    kill_flask_backend()  # Clean up the process
    
    # Collect any remaining output
    while not output_queue.empty():
        line = output_queue.get_nowait().strip()
        if line:
            captured_output.append(line)

    error_message = "Error: Flask backend failed to start within the 30-second timeout."
    if captured_output:
        error_message += "\n\nCaptured Output:\n" + "\n".join(captured_output)
    
    print(error_message)
    return error_message

def kill_flask_backend():
    """Gracefully terminates the Flask backend process."""
    global _flask_process
    if _flask_process:
        print(f"Terminating Flask backend process with PID: {_flask_process.pid}")
        _flask_process.terminate()
        try:
            _flask_process.wait(timeout=5)
            print("Flask backend process terminated.")
        except subprocess.TimeoutExpired:
            print("Flask backend process did not terminate gracefully, killing it.")
            _flask_process.kill()
        _flask_process = None
