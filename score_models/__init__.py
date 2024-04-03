import os
import importlib


def import_subpackage(backend,  subpackage_name):
    full_package_name = f"score_models.{backend}.{subpackage_name}"
    subpackage = importlib.import_module(full_package_name)
    globals()[subpackage_name] = subpackage


PREFERRED_BACKEND = os.environ.get('SCORE_MODELS_BACKEND', None)
JAX_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Select backend based on preference and availability
if PREFERRED_BACKEND == 'jax' and JAX_AVAILABLE:
    from .jax import *  # Import JAX version
    from .jax import layers
    backend = 'jax'

elif PREFERRED_BACKEND == 'torch' and TORCH_AVAILABLE:
    from .torch import *  # Import PyTorch version
    from .torch import layers
    backend = 'torch'
    
elif JAX_AVAILABLE:
    from .jax import *  # Fallback to JAX if available
    from .jax import layers
    backend = 'jax'
    
elif TORCH_AVAILABLE:
    from .torch import *  # Fallback to PyTorch if available
    from .torch import layers
    backend = 'torch'
    
else:
    raise ImportError("Neither JAX nor PyTorch is available. Install one of them to use this package.")

submodules = [
        'layers', 
        'ode', 
        'sde', 
        'utils', 
        'dsm', 
        'ema', 
        'sliced_score_matching', 
        'slic', 
        'definitions',
        'architectures']
for submodule in submodules:
    import_subpackage(backend, submodule)

