import torch
from torch.func import vjp, vmap

__all__ = ["sliced_score_matching_loss"]

# Kept here for reference, but not currently used
def time_weighted_sliced_score_matching_loss(model, x, t, lambda_t, cotangent_vectors=1,  noise_type="rademacher"):
    """
    Score matching loss with the Hutchinson trace estimator trick. See Theorem 1 of
    Hyvärinen (2005). Estimation of Non-Normalized Statistical Models by Score Matching,
    (https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf).

    We implement an unbiased estimator of this loss with reduced variance reported in
    Y. Song et al. (2019). A Scalable Approach to Density and Score Estimation
    (https://arxiv.org/abs/1905.07088).

    Inspired from the official implementation of Sliced Score Matching at https://github.com/ermongroup/sliced_score_matching
    We also implement the weighting scheme for NCSN (Song & Ermon 2019 https://arxiv.org/abs/1907.05600)
    """
    if noise_type not in ["gaussian", "rademacher"]:
        raise ValueError("noise_type has to be either 'gaussian' or 'rademacher'")
    B, *D = x.shape

    # sample cotangent vectors
    z = torch.randn(cotangent_vectors, B, *D)
    if noise_type == 'rademacher':
        z = z.sign()
    score, vjp_func = vjp(lambda x: model(t, x), x)
    trace_estimate = (z * vmap(vjp_func)(z)[0]).mean(0).flatten(1).sum(1)
    loss = (lambda_t(t) * (0.5 * (score**2).flatten(1).sum(1) + trace_estimate)).mean()
    return loss


def sliced_score_matching_loss(model, x, cotangent_vectors=1,  noise_type="rademacher"):
    """
    Score matching loss with the Hutchinson trace estimator trick. See Theorem 1 of
    Hyvärinen (2005). Estimation of Non-Normalized Statistical Models by Score Matching,
    (https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf).

    We implement an unbiased estimator of this loss with reduced variance reported in
    Y. Song et al. (2019). A Scalable Approach to Density and Score Estimation
    (https://arxiv.org/abs/1905.07088).

    Inspired from the official implementation of Sliced Score Matching at https://github.com/ermongroup/sliced_score_matching
    """
    if noise_type not in ["gaussian", "rademacher"]:
        raise ValueError("noise_type has to be either 'gaussian' or 'rademacher'")
    B, *D = x.shape
    z = torch.randn(cotangent_vectors, B, *D)
    if noise_type == 'rademacher':
        z = z.sign()
    score, vjp_func = vjp(model, x)
    trace_estimate = (z * vmap(vjp_func)(z)[0]).mean(0).flatten(1).sum(1)
    loss = (0.5 * (score**2).flatten(1).sum(1) + trace_estimate).mean()
    return loss

