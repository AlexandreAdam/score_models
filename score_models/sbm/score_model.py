from typing import Union, Optional, Callable
from abc import abstractmethod

from torch.func import grad
from torch import vmap, Tensor
from torch.nn import Module
import numpy as np
import torch

from .base import Base
from ..sde import SDE
from ..losses import dsm
from ..ode import probability_flow_ode, divergence_with_hutchinson_trick
from ..sde import euler_maruyama_method
from ..utils import DEVICE


__all__ = ["ScoreModel"]


class ScoreModel(Base):
    def __init__(
            self, 
            net: Optional[Union[str, Module]] = None,
            sde: Optional[Union[str, SDE]] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            hessian_trace_model: Optional[Union[str, Module]] = None,
            device=DEVICE,
            **hyperparameters
            ):
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        if hessian_trace_model is not None:
            self.hessian_trace_model = hessian_trace_model
        else:
            self.hessian_trace_model = self.divergence
    
    def loss(self, x, *args) -> Tensor:
        return dsm(self, x, *args)

    def reparametrized_score(self, t, x, *args) -> Tensor:
        """
        Numerically stable reparametrization of the score function for the DSM loss. 
        """
        return self.net(t, x, *args)

    def forward(self, t, x, *args):
        """
        Overwrite the forward method to return the score function instead of the model output.
        This also affects the __call__ method of the class, meaning that 
        ScoreModel(t, x, *args) is equivalent to ScoreModel.forward(t, x, *args).
        """
        return self.score(t, x, *args)
    
    def score(self, t, x, *args) -> Tensor:
        _, *D = x.shape
        sigma_t = self.sde.sigma(t).view(-1, *[1]*len(D))
        epsilon_theta = self.reparametrized_score(t, x, *args)
        return epsilon_theta / sigma_t
    
    def ode_drift(self, t, x, *args) -> Tensor:
        """
        Compute the drift of the ODE defined by the score function.
        """
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        f_tilde = f - 0.5 * g**2 * self.score(t, x, *args)
        return f_tilde
    
    def divergence(self, t, x, *args, **kwargs) -> Tensor:
        """
        Compute the divergence of the drift of the ODE defined by the score function.
        """
        return divergence_with_hutchinson_trick(self.ode_drift, t, x, *args, **kwargs)
    
    def hessian_trace(self, t, x, *args, **kwargs) -> Tensor:
        """
        Compute the trace of the Hessian of the score function.
        """
        return self.hessian_trace_model(t, x, *args, **kwargs)
    
    def log_likelihood(self, x, *args, steps, t=0., method="euler", **kwargs) -> Tensor:
        """
        Compute the log likelihood of point x using the probability flow ODE, 
        which makes use of the instantaneous change of variable formula
        developed by Chen et al. 2018 (arxiv.org/abs/1806.07366).
        See Song et al. 2020 (arxiv.org/abs/2011.13456) for usage with SDE formalism of SBM.
        """
        drift = self.ode_drift
        hessian_trace = lambda t, x, *args: self.hessian_trace(t, x, *args, **kwargs)
        # Solve the probability flow ODE up in temperature to time t=1.
        xT, delta_log_p = probability_flow_ode(
                x, 
                *args, 
                steps=steps, 
                drift=drift, 
                hessian_trace=hessian_trace, 
                t0=t, 
                t1=1., 
                method=method)
        # Add the log likelihood of the prior at time t=1.
        log_p = self.sde.prior(x.shape).log_prob(xT) + delta_log_p
        return log_p
    
    def tweedie(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """
        Compute the Tweedie formula for the expectation E[x0 | xt] 
        """
        B, *D = x.shape
        mu = self.sde.mu(t).view(-1, *[1]*len(D))
        sigma = self.sde.sigma(t).view(-1, *[1]*len(D))
        return (x + sigma**2 * self.score(t, x, *args)) / mu
    
    @torch.no_grad()
    def sample(
            self, 
            shape: tuple, # TODO grab dimensions from model hyperparams if available
            steps: int, 
            *args,
            likelihood_score: Optional[Callable] = None,
            guidance_factor: float = 1.,
            stopping_factor: float = np.inf,
            denoise_last_step: bool = True
            ) -> Tensor:
        """
        Sample from the score model by solving the reverse-time SDE using the Euler-Maruyama method.
        """
        batch_size, *D = shape
        likelihood_score = likelihood_score or (lambda t, x: torch.zeros_like(x))
        score = lambda t, x: self.score(t, x, *args) + guidance_factor * likelihood_score(t, x)
        t, x = euler_maruyama_method(
                batch_size=batch_size, 
                dimensions=D, 
                steps=steps, 
                sde=self.sde, 
                score=score,
                stopping_factor=stopping_factor
                )
        if denoise_last_step:
            x = self.tweedie(t, x, *args)
        return x
