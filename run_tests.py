#!/usr/bin/env python3
"""
Test runner for comfyui-workflow-generator

Run this script to execute all tests:
    python run_tests.py

Or run specific test categories:
    python run_tests.py --unit      # Unit tests only
    python run_tests.py --cli       # CLI tests only
    python run_tests.py --executor  # Executor tests only
    python run_tests.py --integration # Integration tests only
"""

import sys
import subprocess
import argparse


def run_tests(test_type=None):
    """Run tests with optional filtering."""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    if test_type:
        if test_type == "unit":
            cmd.extend(["tests/test_generator.py"])
        elif test_type == "cli":
            cmd.extend(["tests/test_cli.py"])
        elif test_type == "executor":
            cmd.extend(["tests/test_executor.py"])
        elif test_type == "integration":
            cmd.extend(["tests/test_integration.py"])
        else:
            print(f"Unknown test type: {test_type}")
            return False
    
    print(f"Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run tests for comfyui-workflow-generator")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--cli", action="store_true", help="Run CLI tests only")
    parser.add_argument("--executor", action="store_true", help="Run executor tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    
    args = parser.parse_args()
    
    if args.unit:
        success = run_tests("unit")
    elif args.cli:
        success = run_tests("cli")
    elif args.executor:
        success = run_tests("executor")
    elif args.integration:
        success = run_tests("integration")
    else:
        success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()