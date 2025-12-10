from typing import Union, Callable

import torch
from torch import Tensor

from . import ScoreModel
from ..sde import SDE
from ..architectures import NullNet


class InterpolatedScoreModel(ScoreModel):
    """
    Smoothly transitions between two score models as a function of t.

    This score model class allows for the interpolation of the scores between
    two models. Can be useful when one model is better at capturing the score in
    the early stages of the SDE and another model is better at capturing the
    score in the later stages of the SDE.

    Args:
        sde: The SDE that the score model is associated with.
        hight_model: The high temperature model.
        lowt_model: The low temperature model.
        beta_scheme: The scheme for the beta parameter. Can be "linear", "square",
            "sqrt", "linear:<i>", "sqrt:<i>", or "sin:<i>". For the "<i>" models
            the ``i`` parameter can be used to scale the ``t`` input to beta making
            the transition happen later.

    """

    def __init__(
        self,
        sde: SDE,
        hight_model: ScoreModel,
        lowt_model: ScoreModel,
        beta_scheme: Union[Callable, str] = "linear",
        epsilon: float = 0.01,
        **kwargs,
    ):
        super().__init__(net=NullNet(isenergy=False), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.hight_model = hight_model
        self.lowt_model = lowt_model
        self.beta_scheme = beta_scheme
        self.epsilon = epsilon

    def beta(self, t: Tensor) -> Tensor:
        T = (t - self.sde.t_min) / (self.sde.t_max - self.sde.t_min)
        if callable(self.beta_scheme):
            return self.beta_scheme(T)
        elif self.beta_scheme == "linear":
            return T
        elif self.beta_scheme == "square":
            return T**2
        elif self.beta_scheme == "sqrt":
            return torch.sqrt(T)
        elif "linear:" in self.beta_scheme:
            return int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T
        elif "sqrt:" in self.beta_scheme:
            return torch.sqrt(int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T)
        elif "sin:" in self.beta_scheme:
            i = int(self.beta_scheme[self.beta_scheme.find(":") + 1 :])
            return torch.where(T > 1 / i, torch.ones_like(T), torch.sin(i * T * torch.pi / 2.0))
        else:
            raise NotImplementedError(f"Unknown beta_scheme {self.beta_scheme}")

    def score(self, t: Tensor, x: Tensor, *args, **kwargs):
        # Compute the weighted score for each model
        beta = torch.clamp(self.beta(kwargs.get("t_a", t)[0]), 0.0, 1.0)
        score = torch.zeros_like(x)
        if beta.item() > self.epsilon:
            score += self.hight_model(t, x, *args, **kwargs) * beta
        if beta.item() < (1 - self.epsilon):
            score += self.lowt_model(t, x, *args, **kwargs) * (1.0 - beta)
        return score
    
    def reparametrized_score(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Reparametrized score is not defined for Interpolated models.")
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("Interpolated models cannot yet be saved.")
    
    def load(self, *args, **kwargs):
        raise NotImplementedError("Interpolated models cannot yet be loaded.")
