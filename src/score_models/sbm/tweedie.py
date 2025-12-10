from typing import Callable, Optional

import torch
from torch.func import grad
from torch import vmap, Tensor

from . import ScoreModel
from ..sde import SDE
from ..architectures import NullNet


class TweedieScoreModel(ScoreModel):
    """
    Convolved likelihood score model using Tweedie's Formula.

    Based on Chung et al. 2022 (doi: 10.48550/arXiv.2209.14687) though we 
    use an annealing prescription to anneal the score. 

    Args:
        sde: The SDE that the score model is associated with.
        prior_model: The model to use for the log likelihood score.
        log_likelihood: The log likelihood function to use. Should
            accept signature ll(sigma_t, x, *args, **kwargs) where the args and
            kwargs will be passed from the forward method.
    """

    def __init__(
        self,
        sde: SDE,
        prior_model: ScoreModel,
        log_likelihood: Optional[Callable] = None,
        log_likelihood_score0: Optional[Callable] = None,
        **kwargs,
    ):
        assert (log_likelihood is None) != (
            log_likelihood_score0 is None
        ), "Either log_likelihood or log_likelihood_score0 must be provided, not both."
        super().__init__(net=NullNet(isenergy=False), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.prior_model = prior_model
        if log_likelihood is not None:
            self.log_likelihood = log_likelihood
        else:
            self.log_likelihood_score0 = log_likelihood_score0

    def tweedie(self, t: Tensor, xt: Tensor, *args, **kwargs):
        sigma_t = self.sde.sigma(t)
        mu_t = self.sde.mu(t)
        x0 = (
            xt + sigma_t.unsqueeze(-1) ** 2 * self.prior_model.score(t, xt, *args, **kwargs)
        ) / mu_t.unsqueeze(-1)
        return x0

    def log_likelihood_score0(self, t: Tensor, x0: Tensor, *args, **kwargs):
        sigma_t = self.sde.sigma(t[0])
        return vmap(grad(lambda x: self.log_likelihood(sigma_t, x, *args, **kwargs).squeeze()))(x0)

    def score(self, t: Tensor, x: Tensor, *args, **kwargs):
        x0, vjp_func = torch.func.vjp(lambda xt: self.tweedie(t, xt, *args, **kwargs), x)
        score0 = self.log_likelihood_score0(t, x0, *args, **kwargs)
        return vjp_func(score0)[0]
    
    def reparametrized_score(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Reparametrized score not implemented for TweedieScoreModel.")
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("Saving not implemented for TweedieScoreModel.")
    
    def load(self, *args, **kwargs):
        raise NotImplementedError("Loading not implemented for TweedieScoreModel.")
