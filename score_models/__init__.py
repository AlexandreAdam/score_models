import os

PREFERRED_BACKEND = os.environ.get('MY_PACKAGE_BACKEND', None)
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
elif PREFERRED_BACKEND == 'torch' and TORCH_AVAILABLE:
    from .torch import *  # Import PyTorch version
    from .torch import layers
elif JAX_AVAILABLE:
    from .jax import *  # Fallback to JAX if available
    from .jax import layers
elif TORCH_AVAILABLE:
    from .torch import *  # Fallback to PyTorch if available
    from .torch import layers
else:
    raise ImportError("Neither JAX nor PyTorch is available. Install one of them to use this package.")

