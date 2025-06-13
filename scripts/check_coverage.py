#!/usr/bin/env python3
"""
Coverage gate checker for Kelpie Carbon v1.

This script enforces a minimum test coverage threshold to prevent regression.
It's designed to be run in CI/CD pipelines and local development.
"""

import sys
import subprocess
import re
from pathlib import Path


def get_coverage_percentage():
    """Run pytest with coverage and extract the total coverage percentage."""
    try:
        # Run pytest with coverage, excluding slow tests for speed
        result = subprocess.run([
            "poetry", "run", "pytest",
            "--cov=src/kelpie_carbon",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-m", "not slow",
            "--tb=no",
            "-q"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode != 0:
            print(f"❌ Tests failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None

        # Extract coverage percentage from output
        output = result.stdout

        # Look for the TOTAL line with coverage percentage
        total_pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
        match = re.search(total_pattern, output)

        if match:
            return int(match.group(1))
        else:
            print("❌ Could not parse coverage percentage from output")
            print("Coverage output:", output)
            return None

    except Exception as e:
        print(f"❌ Error running coverage: {e}")
        return None


def check_coverage_gate(current_coverage, minimum_coverage=23):
    """Check if current coverage meets the minimum threshold."""
    print(f"📊 Current test coverage: {current_coverage}%")
    print(f"🎯 Minimum required coverage: {minimum_coverage}%")

    if current_coverage >= minimum_coverage:
        print(f"✅ Coverage gate PASSED! ({current_coverage}% >= {minimum_coverage}%)")
        return True
    else:
        print(f"❌ Coverage gate FAILED! ({current_coverage}% < {minimum_coverage}%)")
        print(f"   Need to increase coverage by {minimum_coverage - current_coverage} percentage points")
        return False


def main():
    """Main coverage gate checker."""
    print("🔍 Running coverage gate check...")
    print("=" * 50)

    # Get current coverage
    coverage = get_coverage_percentage()

    if coverage is None:
        print("❌ Failed to determine coverage percentage")
        sys.exit(1)

    # Check against gate
    # Baseline is 24%, so gate is set at 23% (baseline - 1%)
    MINIMUM_COVERAGE = 23

    if check_coverage_gate(coverage, MINIMUM_COVERAGE):
        print("\n🎉 Coverage gate check completed successfully!")
        sys.exit(0)
    else:
        print("\n💡 Tips to improve coverage:")
        print("   - Add tests for uncovered modules in src/kelpie_carbon/")
        print("   - Focus on core modules with low coverage")
        print("   - Run 'poetry run pytest --cov=src/kelpie_carbon --cov-report=html' for detailed report")
        print("   - Open htmlcov/index.html to see line-by-line coverage")
        sys.exit(1)


if __name__ == "__main__":
    main()
