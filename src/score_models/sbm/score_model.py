from typing import Union, Optional, Literal, TYPE_CHECKING

from torch import Tensor
from torch.nn import Module
import torch

from .base import Base
from ..sde import SDE
from ..losses import dsm
from ..solver import Solver, ODESolver
from ..utils import DEVICE
if TYPE_CHECKING:
    from score_models import HessianDiagonal


__all__ = ["ScoreModel"]


class ScoreModel(Base):
    def __init__(
        self,
        net: Optional[Union[str, Module]] = None,
        sde: Optional[Union[str, SDE]] = None,
        path: Optional[str] = None,
        checkpoint: Optional[int] = None,
        hessian_diagonal_model: Optional["HessianDiagonal"] = None,
        device=DEVICE,
        **hyperparameters
    ):
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        if hessian_diagonal_model is not None:
            self.dlogp = hessian_diagonal_model.dlogp
        else:
            self.dlogp = None

    def loss(self, x, *args, step: int) -> Tensor:
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
        epsilon = self.reparametrized_score(t, x, *args, **kwargs)
        return epsilon / sigma_t

    def log_prob(
        self,
        x,
        *args,
        steps,
        t=0.0,
        solver: Literal["euler_ode", "rk2_ode", "rk4_ode"] = "euler_ode",
        **kwargs
    ) -> Tensor:
        """
        Compute the log likelihood of point x using the probability flow ODE,
        which makes use of the instantaneous change of variable formula
        developed by Chen et al. 2018 (arxiv.org/abs/1806.07366).
        See Song et al. 2020 (arxiv.org/abs/2011.13456) for usage with SDE formalism of SBM.
        """
        B, *D = x.shape
        solver = ODESolver(self, solver=solver, **kwargs)
        # Solve the probability flow ODE up in temperature to time t=1.
        xT, dlogp = solver(
            x, *args, steps=steps, forward=True, t_min=t, **kwargs, return_logp=True, dlogp=self.dlogp
        )
        # add boundary condition PDF probability
        logp = self.sde.prior(D).log_prob(xT) + dlogp
        return logp

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        steps: int,
        *args,
        solver: Literal[
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

        solver = Solver(self, solver=solver, **kwargs)
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

    @torch.no_grad()
    def denoise(
        self,
        t: Tensor,
        xt: Tensor,
        steps: int,
        *args,
        solver: Literal[
            "em_sde", "rk2_sde", "rk4_sde", "euler_ode", "rk2_ode", "rk4_ode"
        ] = "em_sde",
        progress_bar: bool = True,
        denoise_last_step: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Sample from the score model by solving the reverse-time SDE using the Euler-Maruyama method.

        The initial condition is provided as xt at time t.

        """
        x0 = Solver(self, solver=solver, **kwargs)(
            xt,
            *args,
            t_max=t,
            steps=steps,
            forward=False,
            progress_bar=progress_bar,
            denoise_last_step=denoise_last_step,
            **kwargs
        )
        return x0

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute the Tweedie formula for the expectation E[x0 | xt]
        """
        B, *D = x.shape
        mu = self.sde.mu(t).view(-1, *[1] * len(D))
        sigma = self.sde.sigma(t).view(-1, *[1] * len(D))
        return (x + sigma**2 * self.score(t, x, *args, **kwargs)) / mu
