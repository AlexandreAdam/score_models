import torch
import torch.nn as nn
from torch.func import grad
from torch import vmap


class TweedieScoreModel(nn.Module):
    """
    Convolved likelihood score model using Tweedie's Formula.

    Based on Chung et al. 2022 (doi: 10.48550/arXiv.2209.14687) though we use
    the jacobian to properly propagate the score. Uses the score of the expected
    value as an approximation of the expectation of the score.

    Args:
        sde: The SDE that the score model is associated with.
        model: The model to use for the log likelihood score.
        log_likelihood: The log likelihood function to use.
    """

    def __init__(
        self,
        sde,
        model,
        log_likelihood,
    ):
        super().__init__()
        self.sde = sde
        self.model = model
        self.log_likelihood = log_likelihood
        self.hyperparameters = {"nn_is_energy": True}

    def tweedie(self, t, xt):
        sigma_t = self.sde.sigma(t)
        t_mu = self.sde.mu(t)
        x0 = (xt + sigma_t.unsqueeze(-1) ** 2 * self.model.score(t, xt)) / t_mu.unsqueeze(-1)
        return x0

    def log_likelihood_score0(self, t, x0):
        sigma_t = self.sde.sigma(t[0])
        return vmap(grad(lambda x: self.log_likelihood(sigma_t, x).squeeze()))(x0)

    def log_likelihood_score(self, t, xt):
        x0, vjp_func = torch.func.vjp(lambda x: self.tweedie(t, x), xt)
        score0 = self.log_likelihood_score0(t, x0)
        return vjp_func(score0)

    def forward(self, t, xt, *args, **kwargs):

        sigma_t = self.sde.sigma(t[0])
        (scores,) = self.log_likelihood_score(t, xt)
        return scores * sigma_t
