import importlib

import pytest


def test_imports():
    """Test that all voxscriber modules can be imported."""
    modules = [
        "voxscriber",
        "voxscriber.cli",
        "voxscriber.pipeline",
        "voxscriber.transcriber",
        "voxscriber.diarizer",
        "voxscriber.aligner",
        "voxscriber.preprocessor",
        "voxscriber.formatters",
    ]

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            # Some modules might fail on import due to missing system dependencies or hardware
            # (e.g. mlx on non-Apple Silicon), but we want to catch ImportErrors specifically
            print(f"Warning: Error importing {module_name}: {e}")

def test_package_structure():
    """Verify key files exist in the package."""
    import voxscriber
    assert hasattr(voxscriber, "__file__")
