from typing import Union, Optional, Literal

from torch import Tensor
from torch.nn import Module
import torch

from .base import Base
from ..sde import SDE
from ..losses import dsm
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
        self.hessian_trace_model = hessian_trace_model

    def loss(self, x, *args) -> Tensor:
        return dsm(self, x, *args)

    def reparametrized_score(self, t, x, *args, **kwargs) -> Tensor:
        """
        Numerically stable reparametrization of the score function for the DSM loss.
        """
        return self.net(t, x, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        """
        Overwrite the forward method to return the score function instead of the model output.
        This also affects the __call__ method of the class, meaning that
        ScoreModel(t, x, *args) is equivalent to ScoreModel.forward(t, x, *args).
        """
        return self.score(t, x, *args, **kwargs)

    def score(self, t, x, *args, **kwargs) -> Tensor:
        _, *D = x.shape
        sigma_t = self.sde.sigma(t).view(-1, *[1] * len(D))
        epsilon_theta = self.reparametrized_score(t, x, *args, **kwargs)
        return epsilon_theta / sigma_t

    def log_likelihood(
        self,
        x,
        *args,
        steps,
        t=0.0,
        method: Literal["euler_ode", "rk2_ode", "rk4_ode"] = "euler_ode",
        **kwargs
    ) -> Tensor:
        """
        Compute the log likelihood of point x using the probability flow ODE,
        which makes use of the instantaneous change of variable formula
        developed by Chen et al. 2018 (arxiv.org/abs/1806.07366).
        See Song et al. 2020 (arxiv.org/abs/2011.13456) for usage with SDE formalism of SBM.
        """

        solver = self.get_solver(method, ["ode"], **kwargs)
        # Solve the probability flow ODE up in temperature to time t=1.
        _, log_p = solver(x, *args, steps=steps, forward=True, t_min=t, **kwargs, get_logP=True)

        return log_p

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,  # TODO grab dimensions from model hyperparams if available
        steps: int,
        *args,
        method: Literal[
            "em_sde", "rk2_sde", "rk4_sde", "euler_ode", "rk2_ode", "rk4_ode"
        ] = "em_sde",
        progress_bar: bool = True,
        denoise_last_step: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Sample from the score model by solving the reverse-time SDE using the Euler-Maruyama method.

        The initial condition is sample from the high temperature prior at time t=T.
        To denoise a sample from some time t, use the denoise or tweedie method instead.

        """
        B, *D = shape

        solver = self.get_solver(method, **kwargs)
        xT = self.sde.prior(D).sample([B])
        x0 = solver(
            xT,
            *args,
            steps=steps,
            forward=False,
            progress_bar=progress_bar,
            denoise_last_step=denoise_last_step,
            **kwargs
        )

        return x0

    def get_solver(self, method, category=["ode", "sde"], **kwargs):
        msg = "Method not supported, should be one of"
        if "sde" in category:
            msg += " 'em_sde', 'rk2_sde', 'rk4_sde'"
        if "ode" in category:
            msg += " 'euler_ode', 'rk2_ode', 'rk4_ode'"
        for cat in category:
            if cat in method.lower():
                break
        else:
            raise ValueError(msg)
        if method.lower() == "em_sde":
            solver = EM_SDE(self, **kwargs)
        elif method.lower() == "rk2_sde":
            solver = RK2_SDE(self, **kwargs)
        elif method.lower() == "rk4_sde":
            solver = RK4_SDE(self, **kwargs)
        elif method.lower() == "euler_ode":
            solver = Euler_ODE(self, **kwargs)
        elif method.lower() == "rk2_ode":
            solver = RK2_ODE(self, **kwargs)
        elif method.lower() == "rk4_ode":
            solver = RK4_ODE(self, **kwargs)
        else:
            raise ValueError(msg)
        return solver
