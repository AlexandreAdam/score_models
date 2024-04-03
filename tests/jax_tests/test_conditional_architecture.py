from score_models.jax import NCSNpp, DDPM, MLP

import jax.numpy as jnp
from flax import linen as nn
import jax

import pytest

def test_discrete_timelike_conditional():
    nf = 32
    key = jax.random.PRNGKey(0)

    net = NCSNpp(
        nf=nf,
        ch_mult=(1, 1),
        condition=["discrete_timelike"],
        condition_num_embedding=[10],
        key=key
    )

    B = 10
    rng = jax.random.PRNGKey(0)
    c = jax.random.randint(rng, shape=(B,), minval=0, maxval=10)
    x = jax.random.normal(rng, shape=(B, 1, 8, 8))
    t = jax.random.uniform(rng, shape=(B,))

    out = net(t, x, c)

    assert out.shape == x.shape
    assert net.condition_embedding_layers[0](c).shape == (B, nf)

def test_continuous_timelike_conditional():
    nf = 32

    net = NCSNpp(
        nf=nf,
        ch_mult=(1, 1),
        condition=["continuous_timelike"]
    )

    B = 10
    rng = jax.random.PRNGKey(0)
    c = jax.random.normal(rng, shape=(B,))
    x = jax.random.normal(rng, shape=(B, 1, 8, 8))
    t = jax.random.uniform(rng, shape=(B,))

    out = net(t, x, c)

    assert out.shape == x.shape
    assert net.condition_embedding_layers[0](c).shape == (B, nf)

def test_continuous_input_conditional():
    nf = 32
    C_cond = 3

    net = NCSNpp(
        nf=nf,
        ch_mult=(1, 1),
        condition=["input"],
        condition_input_channels=3
    )

    B = 10
    rng = jax.random.PRNGKey(0)
    c = jax.random.normal(rng, shape=(B, C_cond, 8, 8))
    x = jax.random.normal(rng, shape=(B, 1, 8, 8))
    t = jax.random.uniform(rng, shape=(B,))

    out = net(t, x, c)

    assert out.shape == x.shape

def test_vector_condition():
    nf = 32
    C_cond = 3

    net = NCSNpp(
        nf=nf,
        ch_mult=(1, 1),
        condition=["vector"],
        condition_vector_channels=3
    )

    B = 10
    rng = jax.random.PRNGKey(0)
    c = jax.random.normal(rng, shape=(B, C_cond))
    x = jax.random.normal(rng, shape=(B, 1, 8, 8))
    t = jax.random.uniform(rng, shape=(B,))

    out = net(t, x, c)

    assert out.shape == x.shape

def test_mix_condition_type():
    nf = 32
    C_cond = 3

    net = NCSNpp(
        nf=nf,
        ch_mult=(1, 1),
        condition=["input", "discrete_timelike", "continuous_timelike", "continuous_timelike"],
        condition_input_channels=3,
        condition_num_embedding=(15,),
    )

    B = 10
    rng = jax.random.PRNGKey(0)
    c_input = jax.random.normal(rng, shape=(B, C_cond, 8, 8))
    c_discrete = jax.random.randint(rng, shape=(B,), minval=0, maxval=15)
    c_cont1 = jax.random.normal(rng, shape=(B,))
    c_cont2 = jax.random.normal(rng, shape=(B,))
    x = jax.random.normal(rng, shape=(B, 1, 8, 8))
    t = jax.random.uniform(rng, shape=(B,))

    out = net(t, x, c_input, c_discrete, c_cont1, c_cont2)

    assert out.shape == x.shape

def test_conditional_architecture_raising_errors():
    nf = 32

    with pytest.raises(ValueError):
        net = NCSNpp(
            nf=nf,
            ch_mult=(1, 1),
            condition=["discrete_timelike"],
        )

    with pytest.raises(ValueError):
        net = NCSNpp(
            nf=nf,
            ch_mult=(1, 1),
            condition=["discrete_timelike"],
            condition_num_embedding=15
        )

    with pytest.raises(ValueError):
        net = NCSNpp(
            nf=nf,
            ch_mult=(1, 1),
            condition=["input"],
        )

    with pytest.raises(ValueError):
        net = NCSNpp(
            nf=nf,
            ch_mult=(1, 1),
            condition="input",
        )
