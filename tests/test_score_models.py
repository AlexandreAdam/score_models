from score_models.architectures.ddpm import DDPM
import torch
from score_models.utils import load_architecture
from score_models import ScoreModel, EnergyModel
from score_models.architectures import MLP, NCSNpp
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
    ll = score.log_likelihood(x, ode_steps=10, verbose=1, epsilon=1e-5)
    print(ll)
    assert ll.shape == torch.Size([3])

def test_score_at_zero_t():
    net = MLP(dimensions=2)
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    print(score.sde)
    x = torch.randn(3, 2)
    ll, vjp_func = torch.func.vjp(lambda x: score.log_likelihood(x, ode_steps=10, epsilon=1e-5), x)
    grad = vjp_func(torch.ones_like(ll))
    print(grad)

def test_sample_fn():
    net = NCSNpp(1, nf=8, ch_mult=(2, 2))
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    score.sample(5, shape=[1, 16, 16], steps=10)

    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    score.sample(5, shape=[1, 16, 16], steps=10)

    

    
