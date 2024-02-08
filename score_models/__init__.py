# Check if JAX or PyTorch is available
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
    from .jax_dir import *  # Import JAX version
elif TORCH_AVAILABLE:
    from .torch_dir import *  # Import PyTorch version
else:
    raise ImportError("Neither JAX nor PyTorch is available. Install one of them to use this package.")

