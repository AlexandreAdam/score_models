from typing import Union
from jaxtyping import Array, PRNGKeyArray
import jax.numpy as jnp
from jax import random

def denoising_score_matching(key: PRNGKeyArray, score_model: Union["ScoreModel", "EnergyModel"], samples: Array, *args: list[Array]):
    B, *D = samples.shape
    sde = score_model.sde
    keyz, keyt = random.split(key)
    z = random.normal(keyz, samples.shape)
    t = random.uniform(keyt, (B,)) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, samples)
    return jnp.sum((z + score_model.model(t, mean + sigma * z, *args)) ** 2) / B

