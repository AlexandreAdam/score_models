from typing import Union, Optional, Callable

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from .sde import SDE
from ..utils import DEVICE

__all__ = ["euler_maruyama_method"]

def euler_maruyama_method(
        batch_size: int,
        dimensions: tuple[int],
        steps: int, 
        sde: SDE,
        score: Optional[Callable[Tensor, Tensor]] = None,
        T: Optional[Union[Tensor, float]] = None,
        epsilon: Optional[float] = None ,
        guidance_factor: float = 1., 
        stopping_factor: float = 1e2,
        denoise_last_step: bool = True,
        device = DEVICE
        ) -> Tensor:
        """
        An Euler-Maruyama integration of an SDE specified by the score function.
        
        Args:
            batch_size: Number of samples to draw
            dimensions: Shape of the tensor to sample
            steps: Number of Euler-Maruyam steps to perform
            score: Score function of the reverse-time SDE
            likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
            stopping_factor: When magnitude of the score is larger than stopping_factor * sqrt(D), stop the sampling
        """
        B = batch_size
        D = dimensions
        T = T or sde.T
        epsilon = epsilon or sde.epsilon
        score = score or (lambda t, x: torch.zeros_like(x))
         
        x = sde.prior(D).sample([B]).to(device)
        dt = -(T - epsilon) / steps
        t = torch.ones(B).to(device) * T
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
