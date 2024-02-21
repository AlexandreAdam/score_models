try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Dynamically load modules based on availability
if JAX_AVAILABLE:
    from .jax_module import *  # Import JAX version
    from .jax_module import layers
elif TORCH_AVAILABLE:
    from .torch_module import *  # Import PyTorch version
    from .torch_module import layers
else:
    raise ImportError("Neither JAX nor PyTorch is available. Install one of them to use this package.")

