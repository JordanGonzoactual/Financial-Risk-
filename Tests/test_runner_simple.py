#!/usr/bin/env python
"""
Simple test runner for individual test files.
This script sets up the Python path correctly and runs individual test files.
"""

import os
import sys
import subprocess

def setup_python_path():
    """Add project root to Python path."""
    # Get the project root (parent of Tests directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Add to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Also set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if project_root not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = project_root
    
    print(f"✅ Python path set up. Project root: {project_root}")
    return project_root

def run_test_file(test_file_path):
    """Run a specific test file using pytest."""
    if not os.path.exists(test_file_path):
        print(f"❌ Test file not found: {test_file_path}")
        return False
    
    print(f"🧪 Running test file: {test_file_path}")
    
    # Use pytest to run the test file
    cmd = [sys.executable, '-m', 'pytest', test_file_path, '-v', '--tb=short']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Tests passed!")
            return True
        else:
            print(f"❌ Tests failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_runner_simple.py <test_file_path>")
        print("Example: python test_runner_simple.py test_feature_engineering/test_schema_validator.py")
        sys.exit(1)
    
    # Set up Python path
    project_root = setup_python_path()
    
    # Get test file path
    test_file = sys.argv[1]
    
    # If relative path, make it relative to Tests directory
    if not os.path.isabs(test_file):
        test_file = os.path.join(os.path.dirname(__file__), test_file)
    
    # Run the test
    success = run_test_file(test_file)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
