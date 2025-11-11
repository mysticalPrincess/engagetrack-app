"""
Test runner script
Run all tests or specific test suites
"""

import sys
import pytest

def main():
    """Run tests with pytest"""
    
    print("=" * 60)
    print("EngageTrack Test Suite")
    print("=" * 60)
    
    # Default arguments
    args = [
        'tests/',
        '-v',
        '--tb=short',
        '--color=yes'
    ]
    
    # Add command line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run pytest
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed with exit code: {exit_code}")
    print("=" * 60)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
