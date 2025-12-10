from typing import Callable
import torch
from .sbm.score_model import ScoreModel

__all__ = ["PosteriorScoreModel"]

class PosteriorScoreModel(ScoreModel):
    def __init__(self, prior_model: ScoreModel, likelihood_score: Callable):
        # Reuse the neural network, SDE, and hyperparameters from the provided model.
        hp = prior.hyperparameters
        hp.pop("path", None)
        super().__init__(net=prior.net, device=prior.device, **hp)
        self.likelihood_score = likelihood
        self.prior = prior_model

    def score(self, t, x, *args, **kwargs) -> torch.Tensor:
        prior_score = super().score(t, x, *args, **kwargs)
        likelihood_score = self.likelihood_score(t, x)
        return prior_score + likelihood_score

