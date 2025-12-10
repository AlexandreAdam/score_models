from typing import Union, Optional, Literal, Callable, TYPE_CHECKING

from torch import Tensor
from torch.nn import Module
from functools import lru_cache
import torch

from .score_model import ScoreModel
from ..sde import SDE
from ..losses import karras_dsm
from ..solver import Solver, ODESolver
from ..utils import DEVICE
if TYPE_CHECKING:
    from score_models import HessianDiagonal


__all__ = ["EDMScoreModel"]


@lru_cache
def check_that_net_can_return_logvar(net_class_name, raise_error=True) -> bool:
    if not net_class_name in ["MLPv2", "EDMv2Net"]:
        if raise_error:
            raise ValueError(f"Model {net_class_name} does not support return_logvar=True.")


class EDMScoreModel(ScoreModel):
    def __init__(
        self,
        net: Optional[Union[str, Module]] = None,
        sde: Optional[Union[str, SDE]] = None,
        path: Optional[str] = None,
        checkpoint: Optional[int] = None,
        # hessian_diagonal_model: Optional["HessianDiagonal"] = None,
        device=DEVICE,
        **hyperparameters
    ):
        # Hessian Diagonal model is not supported for EDM models
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        self.hyperparameters["formulation"] = "edm"

    def loss(self, x, *args) -> Tensor:
        if hasattr(self, "adaptive_loss"):
            return karras_dsm(self, x, *args, adaptive_loss=self.adaptive_loss)
        else:
            return karras_dsm(self, x, *args)
    
    def sample_noise_level(self, B: int) -> Tensor:
        return self._uniform_noise_level_distribution(B)

    def reparametrized_score(self, t, x, *args, return_logvar=False, **kwargs) -> Tensor:
        """
        In this formulation, reparametrized score is F_theta(t, x)
        """
        B, *D = x.shape
        c_in = self.sde.c_in(t).view(B, *[1]*len(D))
        return self.net(t, c_in * x, *args, return_logvar=return_logvar, **kwargs)
    
    def score(self, t, x, *args, return_logvar=False, **kwargs) -> Tensor:
        """
        Score function is defined through Tweedie's formula and the preconditioned denoiser. 
        For the VP, we use the edm_scale function, one can look at equation 186 in 
        Karras et al. (2022) for the EDM formulation. 
        This works for all SDEs, including VE.
        """
        B, *D = x.shape
        x0 = self.preconditioned_denoiser(t, x, *args, return_logvar=return_logvar, **kwargs) # Estimate of E[x0 | xt]
        if return_logvar:
            check_that_net_can_return_logvar(self.net.__class__.__name__)
            x0, logvar = x0
        sigma = self.sde.sigma(t).view(B, *[1]*len(D))
        var = sigma**2
        if return_logvar:
            return (x0 - x) / var, logvar
        return (x0 - x) / var

    def preconditioned_denoiser(self, t, x: Tensor, *args, return_logvar=False, **kwargs) -> Tensor:
        B, *D = x.shape
        # Broadcast the coefficients to the shape of x
        F_theta = self.reparametrized_score(t, x, *args, return_logvar=return_logvar, **kwargs)
        if return_logvar:
            check_that_net_can_return_logvar(self.net.__class__.__name__)
            F_theta, logvar = F_theta
        c_out = self.sde.c_out(t).view(B, *[1]*len(D))
        c_skip = self.sde.c_skip(t).view(B, *[1]*len(D))
        if return_logvar:
            return c_skip * x + c_out * F_theta, logvar
        return c_skip * x + c_out * F_theta
    
    def _uniform_noise_level_distribution(self, B: int) -> Tensor:
        return torch.rand(B, device=self.device) * (self.sde.T - self.sde.epsilon) + self.sde.epsilon
    
    def _normal_noise_level_distribution(self, B: int) -> Tensor:
        """
        Sample noise level from a log-Normal distribution, 
        then compute the corresponding time-index for this noise level.
        This is the recommended setting for the EDM formulation.
        """
        log_sigma = torch.randn(B, device=self.device) * self.log_sigma_std + self.log_sigma_mean
        sigma = 10**log_sigma
        t = self.sde.sigma_inverse(sigma)
        return t

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.preconditioned_denoiser(t, x, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        return self.score(t, x, *args, **kwargs)
    
    def fit(
            self,
            *args,
            sample_noise_level_function: Optional[Callable] = None,
            noise_level_distribution: Literal["uniform", "normal"] = "uniform",
            log_sigma_mean: Optional[float] = None,
            log_sigma_std: Optional[float] = None,
            adaptive_loss: bool = False,
            **kwargs
            ):
        if sample_noise_level_function is None:
            if noise_level_distribution == "uniform":
                print("Samplng noise level from a Uniform in [epsilon, T]")
                self.sample_noise_level = self._uniform_noise_level_distribution
            elif noise_level_distribution == "normal":
                if log_sigma_mean is None or log_sigma_std is None:
                    raise ValueError("You must provide log_sigma_mean and log_sigma_std when using normal distribution, in base 10.")
                print(f"Sampling noise level from log-Normal with mean log sigma = {log_sigma_mean} and standard deviation {log_sigma_std} (base 10)")
                self.log_sigma_mean = log_sigma_mean
                self.log_sigma_std = log_sigma_std
                self.sample_noise_level = self._normal_noise_level_distribution
            else:
                raise ValueError(f"Sampling distribution {noise_level_distribution} is not recognized. Choose between 'uniform' and 'normal'.")
        else:
            print(f"Using custom function {sample_noise_level_function.__name__} to sample noise level ") 
            self.sample_noise_level = sample_noise_level_function
        
        self.adaptive_loss = adaptive_loss
        if adaptive_loss:
            check_that_net_can_return_logvar(self.net.__class__.__name__)
            print("Using adaptive DSM loss with noise level uncertainty estimation.")
        return super().fit(*args, **kwargs)

