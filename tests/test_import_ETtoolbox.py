import sys
import os
import pytest

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def test_import_ETtoolbox():
    print("PYTHONPATH:", sys.path)  # Debug the Python path
    try:
        import ETtoolbox
    except ImportError as e:
        pytest.fail(f"Failed to import ETtoolbox: {e}")
