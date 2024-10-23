from typing import Optional, Literal

from torch import Tensor
from torch.func import vjp
import torch
import os

from .base import Base
from .score_model import ScoreModel
from ..sde import SDE
from ..utils import DEVICE
from ..losses import (
        second_order_dsm, 
        second_order_dsm_meng_variation,
        )
from ..solver import ODESolver

__all__ = ["HessianDiagonal"]


class HessianDiagonal(Base):
    def __init__(
            self,
            score_model: Optional[ScoreModel] = None,
            net: Optional[torch.nn.Module] = None,
            sde: Optional[torch.Tensor] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            device: torch.device = DEVICE,
            loss: Literal["lu", "meng"] = "meng",
            **hyperparameters
            ):
        if isinstance(score_model, ScoreModel):
            sde = score_model.sde
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        # Check if SBM has been loaded, otherwise use the user provided SBM
        if not hasattr(self, "score_model"):
            if not isinstance(score_model, ScoreModel):
                raise ValueError("Must provide a ScoreModel instance to instantiate the HessianDiagonal model.")
            self.score_model = score_model
        if loss.lower() == "lu":
            self._loss = second_order_dsm
        elif loss.lower() == "meng":
            self._loss = second_order_dsm_meng_variation
        else:
            raise ValueError(f"Loss function {loss} is not recognized. Choose between 'Lu' or 'Meng' loss functions.")
        # Make sure ScoreModel weights are frozen (this class does not allow joint optimization for now)
        for p in self.score_model.net.parameters():
            p.requires_grad = False
        print("Score model weights are now frozen. This class does not currently support joint optimization.")
   
    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.diagonal(t, x, *args, **kwargs)
    
    def score(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.score_model.score(t, x, *args, **kwargs)
    
    def loss(self, x: Tensor, *args, step: int) -> Tensor:
        return self._loss(self, x, *args)
    
    def reparametrized_diagonal(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Numerically stable reparametrization of the diagonal of the Hessian for the DSM loss.
        """
        return self.net(t, x, *args, **kwargs)
    
    def diagonal(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Diagonal of the Hessian of the log likelihood with respect to the input x.
        """
        B, *D = x.shape
        sigma_t = self.sde.sigma(t).view(B, *[1]*len(D))
        return (self.net(t, x, *args, **kwargs) - 1) / sigma_t**2
    
    def dlogp(self, t: Tensor, x: Tensor, *args, dt: Tensor, **kwargs):
        """
        Compute the divergence of the probability flow ODE drift function 
        to update the log probability. 
        """
        g = self.sde.diffusion(t, x).squeeze()
        f, vjp_func = vjp(lambda x: self.sde.drift(t, x), x)
        div_f = vjp_func(torch.ones_like(f))[0].flatten(1).sum(1)         # divergence of the drift
        trace_H = self.diagonal(t, x, *args, **kwargs).flatten(1).sum(1)  # trace of the Hessian (divergence of the score)
        return (div_f - 0.5 * g**2 * trace_H) * dt.squeeze()

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
            x, *args, steps=steps, forward=True, t_min=t, **kwargs, return_dlogp=True, dlogp=self.dlogp
        )
        # add boundary condition PDF probability
        logp = self.sde.prior(D).log_prob(xT) + dlogp
        return logp
    
    def save(
            self, 
            path: Optional[str] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            create_path: bool = True,
            ):
        """
        We use the super method to save checkpoints of the hessian diagonal network.
        We need to save a copy of the score model net and hyperparameters 
        in order to reload it correctly. This method add special routines for that.
        """
        super().save(path, optimizer, create_path) # Save Hessian net
        # Create a sub directory for the SBM 
        path = path or self.path
        sbm_path = os.path.join(path, "score_model")
        if not os.path.exists(sbm_path):
            self.score_model.save(sbm_path, create_path=True)
    
    def load(
            self, 
            checkpoint: Optional[int] = None, 
            raise_error: bool = True
            ):
        """
        Super method reloads the HessianDiagonal net.
        Then we load the base score model from the score_model sub-directory.
        """
        super().load(checkpoint, raise_error)
        sbm_path = os.path.join(self.path, "score_model")
        self.score_model = ScoreModel(path=sbm_path)
