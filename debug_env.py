import os
from dotenv import load_dotenv

# --- Debugging Environment Loading ---

# 1. Check current working directory
print(f"Current working directory: {os.getcwd()}")

# 2. List files in the directory to ensure .env is visible
print(f"Files in current directory: {os.listdir('.')}")

# 3. Check environment variables BEFORE loading
print("\n--- Before load_dotenv() ---")
print(f"MLFLOW_DB_USER: '{os.getenv('MLFLOW_DB_USER')}'")
print(f"MLFLOW_DB_PASSWORD: '{os.getenv('MLFLOW_DB_PASSWORD')}'")
print(f"MLFLOW_DB_NAME: '{os.getenv('MLFLOW_DB_NAME')}'")

# 4. Load the .env file
load_dotenv()

# 5. Check environment variables AFTER loading
print("\n--- After load_dotenv() ---")
print(f"MLFLOW_DB_USER: '{os.getenv('MLFLOW_DB_USER')}'")
print(f"MLFLOW_DB_PASSWORD: '{os.getenv('MLFLOW_DB_PASSWORD')}'")
print(f"MLFLOW_DB_NAME: '{os.getenv('MLFLOW_DB_NAME')}'")

# 6. Verify .env file content
print("\n--- .env file content ---")
try:
    with open('.env', 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("Error: .env file not found in the current directory.")
except Exception as e:
    print(f"An error occurred while reading the .env file: {e}")
