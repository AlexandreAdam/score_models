import pytest

def is_jax_installed():
    try:
        import jax
        return True
    except ImportError:
        return False

def is_torch_installed():
    try:
        import torch
        return True
    except ImportError:
        return False

def pytest_collection_modifyitems(config, items):
    if not is_jax_installed():
        skip_jax = pytest.mark.skip(reason="JAX is not installed")
        for item in items:
            if "jax_tests" in str(item.fspath):
                item.add_marker(skip_jax)

    if not is_torch_installed():
        skip_torch = pytest.mark.skip(reason="PyTorch is not installed")
        for item in items:
            if "torch_tests" in str(item.fspath):
                item.add_marker(skip_torch)
