import subprocess
import os
import requests
import time
from dotenv import load_dotenv

def start_mlflow_server():
    """Checks if the MLflow server is running and starts it if not."""
    try:
        requests.get("http://localhost:5000", timeout=5)
        print("MLflow server is already running.")
        return "MLflow server is already running."
    except requests.exceptions.ConnectionError:
        print("MLflow server is not running. Starting server...")

    load_dotenv()

    user = os.getenv('MLFLOW_DB_USER')
    password = os.getenv('MLFLOW_DB_PASSWORD')
    db_name = os.getenv('MLFLOW_DB_NAME')

    cmd = f"mlflow server --backend-store-uri postgresql://{user}:{password}@localhost:5432/{db_name} --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000"

    print(f"Starting MLflow server with command: {cmd}")
    subprocess.Popen(cmd, shell=True)
    print("MLflow server startup process initiated.")
    return "MLflow server startup process initiated."

if __name__ == "__main__":
    start_mlflow_server()
