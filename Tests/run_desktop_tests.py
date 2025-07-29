"""
Desktop Application Test Runner

This script provides a specialized test runner for desktop application tests
with proper environment setup and dependency management.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def setup_test_environment():
    """Setup the test environment for desktop application tests."""
    # Set environment variables for testing
    os.environ['PYTEST_CURRENT_TEST'] = 'true'
    os.environ['TESTING'] = 'true'
    
    # Disable GUI components during testing
    os.environ['DISPLAY'] = ':99'  # Virtual display for Linux
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Qt offscreen platform


def run_desktop_tests(test_markers=None, verbose=False, coverage=False):
    """
    Run desktop application tests with appropriate configuration.
    
    Args:
        test_markers: List of pytest markers to filter tests
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    setup_test_environment()
    
    # Base pytest command
    cmd = [
        sys.executable, '-m', 'pytest',
        'Tests/test_frontend/test_desktop_app.py',
        '--tb=short',
        '--strict-markers'
    ]
    
    # Add verbosity
    if verbose:
        cmd.append('-v')
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            '--cov=frontend.desktop_app',
            '--cov-report=term-missing',
            '--cov-report=html:Tests/coverage_desktop'
        ])
    
    # Add marker filtering
    if test_markers:
        for marker in test_markers:
            cmd.extend(['-m', marker])
    
    # Change to project directory
    os.chdir(project_root)
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test_class(test_class, verbose=False):
    """
    Run a specific test class.
    
    Args:
        test_class: Name of the test class to run
        verbose: Enable verbose output
    """
    setup_test_environment()
    
    cmd = [
        sys.executable, '-m', 'pytest',
        f'Tests/test_frontend/test_desktop_app.py::{test_class}',
        '--tb=short'
    ]
    
    if verbose:
        cmd.append('-v')
    
    os.chdir(project_root)
    
    print(f"Running test class: {test_class}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running test class: {e}")
        return 1


def main():
    """Main entry point for the desktop test runner."""
    parser = argparse.ArgumentParser(description='Run desktop application tests')
    parser.add_argument(
        '--markers', '-m',
        nargs='+',
        help='Pytest markers to filter tests (e.g., unit, integration, gui)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Enable coverage reporting'
    )
    parser.add_argument(
        '--class', '-cl',
        dest='test_class',
        help='Run a specific test class'
    )
    parser.add_argument(
        '--list-markers',
        action='store_true',
        help='List available test markers'
    )
    
    args = parser.parse_args()
    
    if args.list_markers:
        print("Available test markers for desktop tests:")
        print("  desktop: Desktop application tests")
        print("  unit: Unit tests")
        print("  integration: Integration tests")
        print("  gui: GUI-related tests")
        print("  process: Process management tests")
        print("  network: Network-related tests")
        print("  mock_heavy: Tests with extensive mocking")
        print("  slow: Slow running tests")
        return 0
    
    if args.test_class:
        return run_specific_test_class(args.test_class, args.verbose)
    else:
        return run_desktop_tests(args.markers, args.verbose, args.coverage)


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
