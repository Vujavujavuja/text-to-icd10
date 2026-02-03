"""
Setup script using uv for fast dependency installation.

Run this once before using the preprocessing notebook.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        sys.exit(1)
    print(f"\n[SUCCESS] {description} completed!")


def main():
    print("\n" + "="*60)
    print("ICD-10 Coding Assistant - Setup with uv")
    print("="*60)

    # Create virtual environment with uv
    run_command(
        "uv venv",
        "Creating virtual environment"
    )

    # Install development dependencies (includes production deps)
    run_command(
        "uv pip install -r requirements-dev.txt",
        "Installing dependencies with uv"
    )

    # Create necessary directories
    print("\n" + "="*60)
    print("Creating data directories")
    print("="*60)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("[SUCCESS] Directories created")

    # Run utility tests
    print("\n" + "="*60)
    print("Running utility tests")
    print("="*60)
    result = subprocess.run("uv run python test_utilities.py", shell=True)

    if result.returncode != 0:
        print("\n[WARNING] Utility tests failed, but continuing...")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    print("   - Windows: .venv\\Scripts\\activate")
    print("   - Unix: source .venv/bin/activate")
    print("\n2. Run preprocessing notebook:")
    print("   uv run jupyter notebook notebooks/preprocessing.ipynb")
    print("\n3. After preprocessing, start the server:")
    print("   uv run uvicorn app.main:app --reload")
    print()


if __name__ == "__main__":
    main()
