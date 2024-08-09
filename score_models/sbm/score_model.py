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
from ..solver import EM_SDE, RK2_SDE, RK4_SDE, Euler_ODE, RK2_ODE, RK4_ODE
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
        sigma_t = self.sde.sigma(t).view(-1, *[1] * len(D))
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

    def log_likelihood(self, x, *args, steps, t=0.0, method="euler", **kwargs) -> Tensor:
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
            t1=1.0,
            method=method
        )
        # Add the log likelihood of the prior at time t=1.
        log_p = self.sde.prior(x.shape).log_prob(xT) + delta_log_p
        return log_p

    def tweedie(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """
        Compute the Tweedie formula for the expectation E[x0 | xt]
        """
        B, *D = x.shape
        mu = self.sde.mu(t).view(-1, *[1] * len(D))
        sigma = self.sde.sigma(t).view(-1, *[1] * len(D))
        return (x + sigma**2 * self.score(t, x, *args)) / mu

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,  # TODO grab dimensions from model hyperparams if available
        steps: int,
        *args,
        method: str = "EM_SDE",
        progress_bar: bool = True,
        denoise_last_step: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Sample from the score model by solving the reverse-time SDE using the Euler-Maruyama method.
        """
        if method == "EM_SDE":
            solver = EM_SDE(self, **kwargs)
        elif method == "RK2_SDE":
            solver = RK2_SDE(self, **kwargs)
        elif method == "RK4_SDE":
            solver = RK4_SDE(self, **kwargs)
        elif method == "Euler_ODE":
            solver = Euler_ODE(self, **kwargs)
        elif method == "RK2_ODE":
            solver = RK2_ODE(self, **kwargs)
        elif method == "RK4_ODE":
            solver = RK4_ODE(self, **kwargs)
        else:
            raise ValueError(
                "Method not supported, should be one of 'EM_SDE', 'RK2_SDE', 'RK4_SDE', 'Euler_ODE', 'RK2_ODE', 'RK4_ODE'"
            )

        B, *D = shape
        xT = self.sde.prior(D).sample([B])
        x0 = solver.reverse(xT, steps, progress_bar=progress_bar, **kwargs)
        if denoise_last_step:
            x0 = self.tweedie(self.sde.t_min * torch.ones(x0.shape[0], device=DEVICE), x0, *args)
        return x0
