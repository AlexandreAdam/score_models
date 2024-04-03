from jax import vjp
import jax.numpy as jnp
from jax import random
from jaxtyping import PRNGKeyArray
import equinox as eqx


@eqx.filter_value_and_grad
def time_weighted_sliced_score_matching_loss(key: PRNGKeyArray, model, samples, t, lambda_t, n_cotangent_vectors=1,  noise_type="rademacher"):
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
    B, *D = samples.shape
    # duplicate noisy samples across the number of particle for the Hutchinson trace estimator
    samples = jnp.tile(samples, [n_cotangent_vectors, *[1]*len(D)])
    t = jnp.tile(t, [n_cotangent_vectors])

    # sample cotangent vectors
    vectors = random.normal(key, samples.shape)
    if noise_type == 'rademacher':
        vectors = jnp.sign(vectors)
    score, vjp_func = vjp(lambda x: model(t, x), samples)
    trace_estimate = vectors * vjp_func(vectors)[0]
    trace_estimate = jnp.sum(trace_estimate.flatten(1), axis=1)
    loss = (lambda_t(samples, t) * (0.5 * jnp.sum(score.flatten(1)**2, axis=1) + trace_estimate)).mean()
    return loss


@eqx.filter_value_and_grad
def sliced_score_matching_loss(key: PRNGKeyArray, model, samples, n_cotangent_vectors=1,  noise_type="rademacher"):
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
    B, *D = samples.shape
    # duplicate noisy samples across the number of particle for the Hutchinson trace estimator
    samples = jnp.tile(samples, [n_cotangent_vectors, *[1]*len(D)])
    # sample cotangent vectors
    vectors = random.normal(key, samples.shape)
    if noise_type == 'rademacher':
        vectors = jnp.sign(vectors)
    score, vjp_func = vjp(model, samples)
    trace_estimate = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
    loss = (0.5 * jnp.sum(score.flatten(1)**2, axis=1) + trace_estimate).mean()
    return loss

