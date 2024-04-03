import jax.numpy as jnp
from jax import random
from scipy.stats import norm
from jaxtyping import PRNGKeyArray

def variance_scaling(
    scale,
    mode,
    distribution,
    in_axis=1,
    out_axis=0,
    dtype=jnp.float32,
):
    """Ported from Yang Song repo"""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = jnp.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, *, key: PRNGKeyArray):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")

        variance = scale / denominator

        if distribution == "normal":
            return random.normal(key, shape, dtype=dtype) * jnp.sqrt(variance)
        elif distribution == "uniform":
            return (random.uniform(key, shape, dtype=dtype) * 2.0 - 1.0) * jnp.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')

def geometric_series(begin_value, end_value, L):
    r = (end_value / begin_value) ** (1 / L)
    return jnp.concatenate([jnp.array([begin_value * (r ** n)]) for n in range(L)])

def radial_acceptance_probability(gamma, n):
    """
    Technique 2 of Song & Ermon (2020) paper on improved techniques for training SBM
    """
    c = (2 * (n ** (1/2)))
    return norm.cdf(c * (gamma - 1) + 3 * gamma) - norm.cdf(c * (gamma - 1) - 3 * gamma)

