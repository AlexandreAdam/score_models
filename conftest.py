import pytest

def jax_installed():
    try:
        import jax
        return True
    except ImportError:
        return False

def torch_installed():
    try:
        import torch
        return True
    except ImportError:
        return False

def pytest_collection_modifyitems(config, items):
    if not jax_installed():
        skip_jax = pytest.mark.skip(reason="JAX is not installed")
        for item in items:
            print(item.fspath)
            if "jax_tests" in str(item.fspath):
                item.add_marker(skip_jax)

    if not torch_installed():
        skip_torch = pytest.mark.skip(reason="PyTorch is not installed")
        for item in items:
            if "torch_tests" in str(item.fspath):
                item.add_marker(skip_torch)


# def pytest_configure(config):
    # if not jax_installed():
        # config.addinivalue_line("markers", "jax_test: mark test as a JAX test to run only when JAX is installed")
    # if not torch_installed():
        # config.addinivalue_line("markers", "torch_test: mark test as a PyTorch test to run only when PyTorch is installed")

# def pytest_collection_modifyitems(config, items):
    # if not jax_installed():
        # skip_jax = pytest.mark.skip(reason="JAX is not installed")
        # for item in items:
            # if "jax_test" in item.keywords:
                # item.add_marker(skip_jax)

    # if not torch_installed():
        # skip_torch = pytest.mark.skip(reason="PyTorch is not installed")
        # for item in items:
            # if "torch_test" in item.keywords:
                # item.add_marker(skip_torch)
