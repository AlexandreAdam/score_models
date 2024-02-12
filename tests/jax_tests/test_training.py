import jax.numpy as jnp
from jax import random
from score_models import ScoreModel, EnergyModel, MLP, NCSNpp
from score_models.utils import load_architecture
import shutil
import os
import numpy as np
import pytest


class Dataset:
    def __init__(
        self,
        size,
        channels,
        dimensions: list,
        conditioning="None",
        test_input_list=False,
        rng_key=random.PRNGKey(0),
    ):
        self.size = size
        self.channels = channels
        self.dimensions = dimensions
        self.conditioning = conditioning
        self.test_input_list = test_input_list
        self.rng_key = rng_key

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        key, subkey = random.split(self.rng_key)
        if self.test_input_list:
            return (random.normal(subkey, (self.channels, *self.dimensions)),)
        if self.conditioning.lower() == "none":
            return random.normal(subkey, (self.channels, *self.dimensions))
        elif self.conditioning.lower() == "time":
            return random.normal(
                subkey, (self.channels, *self.dimensions)
            ), random.normal(subkey, (1,))
        elif self.conditioning.lower() == "input":
            return random.normal(
                subkey, (self.channels, *self.dimensions)
            ), random.normal(subkey, (self.channels, *self.dimensions))
        elif self.conditioning.lower() == "input_and_time":
            return (
                random.normal(subkey, (self.channels, *self.dimensions)),
                random.normal(subkey, (self.channels, *self.dimensions)),
                random.normal(subkey, (1,)),
            )
        elif self.conditioning.lower() == "time_and_discrete":
            return (
                random.normal(subkey, (self.channels, *self.dimensions)),
                random.normal(subkey, (1,)),
                random.randint(subkey, (1,), 0, 10),
            )
        elif self.conditioning.lower() == "discrete_time":
            return random.normal(subkey, (self.channels, *self.dimensions)), jnp.array(
                np.random.choice(range(10))
            )


def test_multiple_channels_ncsnpp():
    C = 3
    D = 16
    dim = 2
    B = 5
    size = 2 * B
    dataset = Dataset(size, C, [D] * dim)
    net = NCSNpp(nf=8, channels=C, ch_mult=(1, 1))
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_input_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2 * B
    dataset = Dataset(size, C, [D] * dim, conditioning="input")
    net = NCSNpp(nf=8, ch_mult=(1, 1), condition=["input"], condition_input_channels=C)
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_continuous_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2 * B
    dataset = Dataset(size, C, [D] * dim, conditioning="time")
    net = NCSNpp(nf=8, ch_mult=(1, 1), condition=["continuous_timelike"])
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_discrete_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2 * B
    dataset = Dataset(size, C, [D] * dim, conditioning="discrete_time")
    net = NCSNpp(
        nf=8,
        ch_mult=(1, 1),
        condition=["discrete_timelike"],
        condition_num_embedding=(10,),
    )
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_discrete_and_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2 * B
    dataset = Dataset(size, C, [D] * dim, conditioning="time_and_discrete")
    net = NCSNpp(
        nf=8,
        ch_mult=(1, 1),
        condition=["continuous_timelike", "discrete_timelike"],
        condition_num_embedding=(10,),
    )
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)




def local_test_loading_model_and_score_fn():
    # Placeholder for local test only
    path = "/path/to/model/checkpoints"
    model, hparams = load_architecture(path)
    
    score = ScoreModel(checkpoints_directory=path)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 1, 256, 256))
    t = jnp.ones(1)
    score(t, x)

def test_loading_from_string():
    score = ScoreModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2, nn_is_energy=True)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    score = EnergyModel("ncsnpp", sigma_min=1e-2, sigma_max=10, nf=32)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 1, 16, 16))
    t = jnp.ones(1)
    score(t, x)

def test_loading_with_nn():
    net = MLP(dimensions=2)
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    net = MLP(dimensions=2)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    net = MLP(dimensions=2, nn_is_energy=True)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 2))
    t = jnp.ones(1)
    score(t, x)

    net = NCSNpp(nf=32)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (1, 1, 16, 16))
    t = jnp.ones(1)
    score(t, x)

def test_init_score():
    net = MLP(10)
    with pytest.raises(KeyError):
        score = ScoreModel(net)

def test_log_likelihood():
    net = MLP(dimensions=2)
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    print(score.sde)
    x = random.normal(random.PRNGKey(0), (3, 2))
    ll = score.log_likelihood(x, ode_steps=10, verbose=1)
    print(ll)
    assert ll.shape == (3,)

# def test_score_at_zero_t():
    # Placeholder for test function

def test_sample_fn():
    net = NCSNpp(1, nf=8, ch_mult=(2, 2))
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)

    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)

def test_slic_score():
    def forward_model(x):
        return jnp.sum(x, axis=1, keepdims=True) # Function R^C to R
    C = 100
    net = MLP(dimensions=C)
    # Placeholder for SLIC score testing

def test_loading_different_sdes():
    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10, epsilon=1e-3)
    # Placeholder for loading different SDEs

if __name__ == "__main__":
    local_test_loading_model_and_score_fn()

