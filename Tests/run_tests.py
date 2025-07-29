"""
Test runner script for the Financial Risk project.

This script provides a convenient way to run different types of tests
with various configurations and reporting options.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(command, cwd=None):
    """Run a command and return the result."""
    try:
        # Using subprocess.run and letting it inherit stdout/stderr
        # is simpler and more reliable for streaming output.
        result = subprocess.run(
            command,  # command should be a list
            shell=False,
            cwd=cwd,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False # Do not raise exception on non-zero exit codes
        )
        return result
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return None


def install_test_dependencies():
    """Install required test dependencies."""
    print("📦 Installing test dependencies...")
    
    dependencies = [
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0",
        "pytest-html>=3.0.0",
        "pytest-xdist>=2.0.0",
        "pytest-mock>=3.0.0",
        "coverage>=5.0.0",
        "mock>=4.0.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = run_command(f"pip install {dep}")
        if result and result.returncode != 0:
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("✅ Test dependencies installation completed")


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    cmd = [sys.executable, "-m", "pytest", "Tests/", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html:Tests/coverage_html", "--cov-report=term-missing"])
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print(f"\nUnit tests completed with return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ All unit tests passed!")
        else:
            print("❌ Some unit tests failed")
    
    return result


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("🔗 Running integration tests...")
    
    cmd = [sys.executable, "-m", "pytest", "Tests/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print(f"\nIntegration tests completed with return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ All integration tests passed!")
        else:
            print("❌ Some integration tests failed")
    
    return result


def run_performance_tests(verbose=False):
    """Run performance tests."""
    print("⚡ Running performance tests...")
    
    cmd = [sys.executable, "-m", "pytest", "Tests/", "-m", "performance"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print(f"\nPerformance tests completed with return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ All performance tests passed!")
        else:
            print("❌ Some performance tests failed")
    
    return result


def run_desktop_tests(verbose=False, coverage=False):
    """Run desktop application tests."""
    print("🖥️  Running desktop application tests...")
    
    cmd = [sys.executable, "-m", "pytest", "Tests/test_frontend/test_desktop_app.py", "-m", "desktop"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=frontend.desktop_app",
            "--cov-report=html:Tests/coverage_desktop",
            "--cov-report=term-missing"
        ])
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print(f"\nDesktop tests completed with return code: {result.returncode}")
        if result.returncode == 0:
            print("✅ All desktop tests passed!")
        else:
            print("❌ Some desktop tests failed")
    
    return result


def run_specific_component_tests(component, verbose=False):
    """Run tests for a specific component."""
    print(f"🎯 Running tests for {component} component...")
    
    component_map = {
        'backend': 'Tests/test_backend/',
        'frontend': 'Tests/test_frontend/',
        'models': 'Tests/test_models/',
        'feature_engineering': 'Tests/test_feature_engineering/',
        'desktop': 'Tests/test_frontend/test_desktop_app.py'
    }
    
    if component not in component_map:
        print(f"❌ Unknown component: {component}")
        print(f"Available components: {', '.join(component_map.keys())}")
        return None
    
    test_path = component_map[component]
    cmd = [sys.executable, "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print(f"\nComponent tests for '{component}' completed with return code: {result.returncode}")
        if result.returncode == 0:
            print(f"✅ All tests for component '{component}' passed!")
        else:
            print(f"❌ Some tests for component '{component}' failed")
    
    return result


def run_all_tests(verbose=False, coverage=False, parallel=False):
    """Run all tests."""
    print("🚀 Running all tests...")
    
    cmd = [sys.executable, "-m", "pytest", "Tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html:Tests/coverage_html",
            "--cov-report=term-missing",
            "--cov-report=xml:Tests/coverage.xml"
        ])
    
    if parallel:
        cmd.extend(["-n", "auto"])  # Use pytest-xdist for parallel execution
    
    cmd.extend([
        "--tb=short",
        "--durations=10",
        "--junit-xml=Tests/junit.xml"
    ])
    
    start_time = time.time()
    result = run_command(cmd, cwd=get_project_root())
    end_time = time.time()
    
    if result:
        print(f"All tests completed in {end_time - start_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
    
    return result


def generate_test_report():
    """Generate a comprehensive test report."""
    print("📊 Generating test report...")
    
    # Run tests with coverage and HTML report
    cmd = [
        "python", "-m", "pytest", "Tests/",
        "--cov=.",
        "--cov-report=html:Tests/coverage_html",
        "--cov-report=term-missing",
        "--cov-report=xml:Tests/coverage.xml",
        "--html=Tests/report.html",
        "--self-contained-html",
        "--junit-xml=Tests/junit.xml",
        "-v"
    ]
    
    result = run_command(cmd, cwd=get_project_root())
    
    if result:
        print("📋 Test report generated:")
        print("  - HTML Report: Tests/report.html")
        print("  - Coverage HTML: Tests/coverage_html/index.html")
        print("  - Coverage XML: Tests/coverage.xml")
        print("  - JUnit XML: Tests/junit.xml")
    
    return result


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("🔍 Checking test environment...")
    
    project_root = get_project_root()
    
    # Check if Tests directory exists
    tests_dir = project_root / "Tests"
    if not tests_dir.exists():
        print("❌ Tests directory not found")
        return False
    
    # Check for conftest.py
    conftest_file = tests_dir / "conftest.py"
    if not conftest_file.exists():
        print("❌ conftest.py not found")
        return False
    
    # Check for pytest.ini
    pytest_ini = tests_dir / "pytest.ini"
    if not pytest_ini.exists():
        print("⚠️  pytest.ini not found (optional)")
    
    # Check if pytest is installed
    result = run_command("python -m pytest --version")
    if not result or result.returncode != 0:
        print("❌ pytest not installed or not working")
        return False
    
    print("✅ Test environment looks good!")
    return True


def clean_test_artifacts():
    """Clean up test artifacts and cache files."""
    print("🧹 Cleaning test artifacts...")
    
    project_root = get_project_root()
    
    # Directories and files to clean
    cleanup_paths = [
        "Tests/__pycache__",
        "Tests/.pytest_cache",
        "Tests/coverage_html",
        "Tests/coverage.xml",
        "Tests/junit.xml",
        "Tests/report.html",
        ".coverage",
        "**/__pycache__",
        "**/.pytest_cache"
    ]
    
    import shutil
    import glob
    
    for pattern in cleanup_paths:
        if pattern.startswith("**"):
            # Handle recursive patterns
            for path in glob.glob(str(project_root / pattern), recursive=True):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    else:
                        os.remove(path)
                        print(f"Removed file: {path}")
                except Exception as e:
                    print(f"Warning: Could not remove {path}: {e}")
        else:
            path = project_root / pattern
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    else:
                        path.unlink()
                        print(f"Removed file: {path}")
            except Exception as e:
                print(f"Warning: Could not remove {path}: {e}")
    
    print("✅ Test artifacts cleaned")


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test runner for Financial Risk project")
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "performance", "all", "component",
            "desktop", "report", "check", "clean", "install-deps"
        ],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--component",
        choices=["backend", "frontend", "models", "feature_engineering", "desktop"],
        help="Specific component to test (use with 'component' command)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(get_project_root())
    
    # Execute the requested command
    if args.command == "install-deps":
        install_test_dependencies()
    
    elif args.command == "check":
        check_test_environment()
    
    elif args.command == "clean":
        clean_test_artifacts()
    
    elif args.command == "unit":
        run_unit_tests(verbose=args.verbose, coverage=args.coverage)
    
    elif args.command == "integration":
        run_integration_tests(verbose=args.verbose)
    
    elif args.command == "performance":
        run_performance_tests(verbose=args.verbose)
    
    elif args.command == "desktop":
        run_desktop_tests(verbose=args.verbose, coverage=args.coverage)
    
    elif args.command == "component":
        if not args.component:
            print("❌ --component argument required for 'component' command")
            sys.exit(1)
        run_specific_component_tests(args.component, verbose=args.verbose)
    
    elif args.command == "all":
        run_all_tests(
            verbose=args.verbose,
            coverage=args.coverage,
            parallel=args.parallel
        )
    
    elif args.command == "report":
        generate_test_report()
    
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    print("🧪 Financial Risk Project Test Runner")
    print("=" * 50)
    main()
