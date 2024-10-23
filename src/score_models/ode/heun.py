from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

__all__ = ["heun_method"]


@torch.no_grad()
def heun_method(
        x: Tensor,
        *args,
        steps: int,
        drift: Callable,
        hessian_trace: Optional[Callable] = None,
        t0: float = 0.,
        t1: float = 1.,
        verbose: int = 0
        ) -> Tuple[Tensor, Tensor]:
    """
    Heun discretisation method of an ODE implicitly defined by a drift function.

    Args:
        x: Initial state
        *args: Additional arguments to pass to the drift function and divergence function.
        steps: Number of steps of integration.
        drift: Update function of the state x.
        hessian_trace: Trace of the Hessian of the log probability. 
            If provided, the trace of the Hessian is integrated alongside the state.
        t0: Initial time of integration, defaults to 0.
        t1: Final time of integration, defaults to 1.
        verbose: If True, display a progress bar.
    """
    B, *D = x.shape
    N = steps
    t = torch.ones([B]).to(x.device) * t0
    dt = (t1 - t0) / N
    
    f = lambda t, x: drift(t, x, *args)
    if hessian_trace is None:
        ht = lambda t, x: 0.
    else:
        ht = lambda t, x: hessian_trace(t, x, *args)

    delta_log_p = 0
    x1 = x.clone()
    for _ in tqdm(range(N), disable=(not verbose)):
        f0 = f(t, x)
        x1 = x + f0 * dt
        f1 = f(t + dt, x1)
        delta_log_p += 0.5 * (ht(t, x) + ht(t + dt, x1)) * dt
        x = x + 0.5 * (f0 + f1) * dt
        t = t + dt
    return x, delta_log_p
