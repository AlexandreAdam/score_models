import torch
from score_models.utils import load_architecture
from score_models import ScoreModel, EnergyModel, SLIC
from score_models.architectures import MLP, NCSNpp, DDPM
from score_models.sde import VESDE, VPSDE, TSVESDE
import pytest


def local_test_loading_model_and_score_fn():
    # local test only
    path = "/home/alexandre/Desktop/Projects/data/score_models/ncsnpp_ct_g_220912024942"
    model, hparams = load_architecture(path)
    
    score = ScoreModel(checkpoints_directory=path)
    print(score.sde)
    x = torch.randn(1, 1, 256, 256)
    t = torch.ones(1)
    score(t, x)


def test_loading_from_string():
    score = ScoreModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, dimensions=2, nn_is_energy=True)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    score = EnergyModel("ncsnpp", sigma_min=1e-2, sigma_max=10, nf=32)
    print(score.sde)
    x = torch.randn(1, 1, 16, 16)
    t = torch.ones(1)
    score(t, x)


def test_loading_with_nn():
    net = MLP(dimensions=2)
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    net = MLP(dimensions=2)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    net = MLP(dimensions=2, nn_is_energy=True)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    net = NCSNpp(nf=32)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 1, 16, 16)
    t = torch.ones(1)
    score(t, x)

def test_init_score():
    net = MLP(10)
    with pytest.raises(KeyError):
        score = ScoreModel(net)

def test_log_likelihood():
    net = MLP(dimensions=2)
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    print(score.sde)
    x = torch.randn(3, 2)
    t = torch.rand(3)
    ll = score.log_likelihood(t, x, ode_steps=10, verbose=1)
    print(ll)
    assert ll.shape == torch.Size([3])

def test_score_at_zero_t():
    net = MLP(dimensions=2)
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    print(score.sde)
    x = torch.randn(3, 2)
    t = torch.rand(3)
    ll, vjp_func = torch.func.vjp(lambda x: score.log_likelihood(t, x, ode_steps=10), x)
    grad = vjp_func(torch.ones_like(ll))
    print(grad)

def test_sample_fn():
    net = NCSNpp(1, nf=8, ch_mult=(2, 2))
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)

    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)

def test_slic_score():
    def forward_model(x):
        return torch.sum(x, dim=1, keepdim=True) # Function R^C to R
    C = 100
    net = MLP(dimensions=C)
    # Check that we can get the score without a forward_model
    score = SLIC(net, beta_min=1e-2, beta_max=10)
    print(score.sde)
    x = torch.randn(3, C)
    t = torch.rand(3)
    s = score(t, x)
    print(s)
    assert s.shape == torch.Size([3, C])

    # Now check slic score
    net = MLP(dimensions=1) # Define SLIC in output space of forward model
    score = SLIC(net, forward_model, beta_min=1e-2, beta_max=10)
    y = forward_model(x)
    print(score.sde)
    x = torch.randn(3, C)
    t = torch.rand(3)
    s = score.slic_score(t, x, y)
    print(s)
    print(s.shape)
    assert s.shape == torch.Size([3, C])
    

def test_loading_different_sdes():
    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10, epsilon=1e-3)
    assert isinstance(score.sde, VPSDE)
    assert score.sde.beta_min == 1e-2
    assert score.sde.beta_max == 10
    assert score.sde.epsilon == 1e-3
    assert score.sde.T == 1

    score = ScoreModel(net, sigma_min=1e-3, sigma_max=1e2)
    assert isinstance(score.sde, VESDE)
    assert score.sde.sigma_min == 1e-3
    assert score.sde.sigma_max == 1e2
    assert score.sde.epsilon == 0
    assert score.sde.T == 1

    score = ScoreModel(net, sigma_min=1e-3, sigma_max=1e2, t_star=0.5, beta=10)
    assert isinstance(score.sde, TSVESDE)
    assert score.sde.sigma_min == 1e-3
    assert score.sde.sigma_max == 1e2
    assert score.sde.epsilon == 0
    assert score.sde.T == 1
    assert score.sde.t_star == 0.5
    assert score.sde.beta == 10



if __name__ == "__main__":
    local_test_loading_model_and_score_fn()
