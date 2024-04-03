from score_models.jax.architectures import NCSNpp, DDPM, MLP
from jax import random


def test_ddpm():
    key = random.PRNGKey(0)
    x = random.normal(key, shape=[1, 1, 32, 32]) * 230
    t = random.normal(key, shape=[1])
    model = DDPM(1, nf=64, ch_mult=(1, 1, 2, 2), key=key)
    model(x=x, t=t)


def test_ddpm_smallnf():
    key = random.PRNGKey(1)
    x = random.normal(key, shape=[1, 1, 32, 32]) * 230
    t = random.normal(key, shape=[1])
    model = DDPM(1, nf=8, ch_mult=(1, 1, 2, 2), key=key)
    model(x=x, t=t)


def test_ncsnpp():
    key = random.PRNGKey(2)
    x = random.normal(key, shape=[1, 1, 32, 32]) * 500
    t = random.normal(key, shape=[1])
    model = NCSNpp(1, dimensions=2, nf=8, ch_mult=(2, 2, 2, 2), num_res_blocks=3, key=key)
    model(x=x, t=t)


def test_ncsnpp1d():
    key = random.PRNGKey(3)
    x = random.normal(key, shape=[1, 1, 256]) * 500
    t = random.normal(key, shape=[1])
    model = NCSNpp(1, dimensions=1, nf=8, ch_mult=(1, 1, 2, 2), attention=True, key=key)
    model(x=x, t=t)


def test_ncsnpp3d():
    key = random.PRNGKey(4)
    x = random.normal(key, shape=[1, 1, 32, 32, 32]) * 500
    t = random.normal(key, shape=[1])
    model = NCSNpp(1, dimensions=3, nf=8, ch_mult=(1, 1, 2, 2), attention=True, key=key)
    model(x=x, t=t)


def test_mlp():
    key = random.PRNGKey(5)
    x = random.normal(key, shape=[10, 10]) * 100
    t = random.normal(key, shape=[10])
    model = MLP(
        dimensions=10,
        units=10,
        layers=3,
        time_embedding_dimensions=16,
        time_branch_layers=2,
        bottleneck=10,
        attention=True,
        key=key,
    )
    model(x=x, t=t)

    key = random.PRNGKey(6)
    x = random.normal(key, shape=[1, 10]) * 100
    t = random.normal(key, shape=[1])
    model = MLP(
        dimensions=10,
        units=10,
        layers=2,
        time_embedding_dimensions=16,
        time_branch_layers=1,
        key=key,
    )
    model(x=x, t=t)
