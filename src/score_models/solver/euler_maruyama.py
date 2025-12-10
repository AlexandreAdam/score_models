"""
Old implementation of the Euler-Maruyama method for sampling from SDEs.
"""
from typing import Union, Optional, Callable

from torch import Tensor
from tqdm import tqdm
from ..sde import SDE
from ..utils import DEVICE
import torch
import numpy as np


__all__ = ["euler_maruyama_method"]

def euler_maruyama_method(
        t: Union[Tensor, float],
        xt: Tensor,
        steps: int, 
        sde: SDE,
        score: Optional[Callable[Tensor, Tensor]] = None,
        epsilon: Optional[float] = None,
        guidance_factor: float = 1., 
        stopping_factor: float = 1e2,
        denoise_last_step: bool = True,
        device = DEVICE
        ) -> Tensor:
        """
        An Euler-Maruyama integration of an SDE specified by the score function.
        
        Args:
            steps: Number of Euler-Maruyam steps to perform
            score: Score function of the reverse-time SDE
            likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
            stopping_factor: When magnitude of the score is larger than stopping_factor * sqrt(D), stop the sampling
        """
        B, *D = xt.shape
        if isinstance(t, float):
            t = torch.ones(B).to(device) * t
        if t.shape[0] == 1:
            t = t.repeat(B).to(device)
        elif not all([t[i].item() == t[0].item() for i in range(B)]):
            raise ValueError("All times must be the same for each batch element, the more general case is not implemented yet.")
        T = t[0].cpu().item()
        epsilon = epsilon or sde.epsilon
        score = score or (lambda t, x: torch.zeros_like(xt))
        x = xt.clone()
        dt = -(T - epsilon) / steps
        for _ in (pbar := tqdm(range(steps))):
            pbar.set_description(f"Euler-Maruyama | t = {t[0].item():.1f} | sigma(t) = {sde.sigma(t)[0].item():.1e}"
                                 f"| x.std() ~ {x.std().item():.1e}")
            g = sde.diffusion(t, x)
            f = sde.drift(t, x) 
            s = score(t, x)
            dw = torch.randn_like(x) * abs(dt)**(1/2)
            x = x + (f - g**2 * s) * dt + g * dw
            t += dt
            # Check for NaNs
            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable: NaN were produced. Stopped sampling.")
                break
            # Check magnitude of the score 
            m = torch.sum(s.flatten(1)**2, dim=1).sqrt()
            if torch.any(m > stopping_factor * np.prod(D)**(1/2)):
                print(f"Diffusion is not stable: magnitude of the score is larger than {stopping_factor} x sqrt(D). Stopped sampling.")
                break
            # Check if t is too small
            if t[0] < epsilon: 
                break
        return t, x
