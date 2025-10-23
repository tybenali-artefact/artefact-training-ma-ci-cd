"""
Test runner script for football prediction tests.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with pytest."""
    print("🧪 Running Football Prediction Tests...")
    print("=" * 50)
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(tests_dir),
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--color=yes",  # Colored output
            "-x",  # Stop on first failure
        ], check=True)
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        print("=" * 50)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        sys.exit(1)


def run_specific_tests(test_pattern=None):
    """Run specific tests based on pattern."""
    print(f"🧪 Running tests matching: {test_pattern}")
    print("=" * 50)
    
    tests_dir = Path(__file__).parent
    
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short", "--color=yes"]
    
    if test_pattern:
        cmd.extend(["-k", test_pattern])
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Tests completed successfully!")
    except subprocess.CalledProcessError as e:
        print("\n❌ Some tests failed!")
        sys.exit(e.returncode)


def run_unit_tests():
    """Run only unit tests."""
    print("🧪 Running Unit Tests...")
    run_specific_tests("unit")


def run_integration_tests():
    """Run only integration tests."""
    print("🧪 Running Integration Tests...")
    run_specific_tests("integration")


def run_fast_tests():
    """Run only fast tests (exclude slow ones)."""
    print("🧪 Running Fast Tests...")
    run_specific_tests("not slow")


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            run_unit_tests()
        elif command == "integration":
            run_integration_tests()
        elif command == "fast":
            run_fast_tests()
        elif command == "help":
            print("Available commands:")
            print("  python run_tests.py           - Run all tests")
            print("  python run_tests.py unit      - Run unit tests only")
            print("  python run_tests.py integration - Run integration tests only")
            print("  python run_tests.py fast      - Run fast tests only")
            print("  python run_tests.py help      - Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'python run_tests.py help' for available commands")
    else:
        run_tests()


if __name__ == "__main__":
    main()
