from typing import Literal, Callable, Optional, Tuple

import torch
from torch import Tensor

from .euler import euler_method
from .heun import heun_method

__all__ = ["probability_flow_ode"]

 
def probability_flow_ode(
        x,
        *args,
        steps: int, 
        drift: Callable,
        hessian_trace: Optional[Callable] = None,
        method: Literal["euler", "heun"] = "euler",
        t0: int = 0,
        t1: int = 1,
        verbose=0) -> Tuple[Tensor, Tensor]:
    """
    The probability flow ODE method for score-based models.
    This method also make use of the instantaneous change of variable formula
    developed by Chen et al. 2018 (arxiv.org/abs/1806.07366) to compute the log probability of the flow.
    See Song et al. 2020 (arxiv.org/abs/2011.13456) for usage with SDE formalism of SBM.
    
    Args:
        x: Initial state
        *args: Additional arguments to pass to the drift function and hessian trace function.
        steps: Number of steps of integration.
        drift: Update function of the state x.
        hessian_trace: Trace of the Hessian of the log probability. 
            If provided, the trace of the Hessian is integrated alongside the state.
        method: Integration method. Either 'euler' or 'heun'.
        t0: Initial time of integration, defaults to 0.
        t1: Final time of integration, defaults to 1.
        verbose: If True, display a progress bar.
    """
    if method == "euler":
        return euler_method(x, *args, steps=steps, drift=drift, hessian_trace=hessian_trace, t0=t0, t1=t1, verbose=verbose)
    elif method == "heun":
        return heun_method(x, *args, steps=steps, drift=drift, hessian_trace=hessian_trace, t0=t0, t1=t1, verbose=verbose)
    else:
        raise NotImplementedError(f"ODE integration method {method} is not implemented. Use 'euler' or 'heun'.")
