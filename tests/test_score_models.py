import torch
from score_models.save_load_utils import initialize_architecture
from score_models import ScoreModel, EnergyModel, SLIC
from score_models.architectures import MLP, NCSNpp, DDPM
from score_models.sde import VESDE, VPSDE, TSVESDE
import pytest


def local_test_loading_model_and_score_fn():
    # local test only
    path = "/home/alexandre/Desktop/Projects/data/score_models/ncsnpp_ct_g_220912024942"
    model, hparams = initialize_architecture(path)

    score = ScoreModel(checkpoints_directory=path)
    print(score.sde)
    x = torch.randn(1, 1, 256, 256)
    t = torch.ones(1)
    score(t, x)


def test_loading_from_string():
    score = ScoreModel("mlp", sigma_min=1e-2, sigma_max=10, channels=2)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, channels=2)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    score = EnergyModel("mlp", sigma_min=1e-2, sigma_max=10, channels =2, nn_is_energy=True)
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
    net = MLP(2)
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    net = MLP(2)
    score = EnergyModel(net, sigma_min=1e-2, sigma_max=10)
    print(score.sde)
    x = torch.randn(1, 2)
    t = torch.ones(1)
    score(t, x)

    net = MLP(2, nn_is_energy=True)
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

def test_sample_method():
    net = NCSNpp(1, nf=8, ch_mult=(2, 2))
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)

    net = DDPM(1, nf=32, ch_mult=(2, 2))
    score = ScoreModel(net, beta_min=1e-2, beta_max=10)
    score.sample(shape=[5, 1, 16, 16], steps=10)


@pytest.mark.parametrize("epsilon", [None, 1e-3, 0.1])
def test_denoise_method(epsilon):
    net = NCSNpp(1, nf=8, ch_mult=(2, 2))
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    B = 5
    t = torch.rand(1) * torch.ones(B)
    x = torch.randn(B, 1, 16, 16)
    score.denoise(t, x, steps=10, epsilon=epsilon)


@pytest.mark.parametrize("anneal_residuals", [True, False])
def test_slic_score(anneal_residuals):
    B = 3
    m = 10
    D = 100

    def forward_model(t, x):
        return x[:, :m]  # Function R^C to R^m

    x = torch.randn(B, D)
    t = torch.rand(B)
    net = MLP(m)  # Define SLIC in output space of forward model (m)
    model = SLIC(forward_model, net, beta_min=1e-2, beta_max=10, anneal_residuals=anneal_residuals)
    y = forward_model(None, x)
    x = torch.randn(B, D)
    t = torch.rand(B)
    s = model(t, y=y, x=x)
    print(s)
    print(s.shape)
    assert s.shape == torch.Size([B, D])


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

    score = ScoreModel(net, sde="tsve", sigma_min=1e-3, sigma_max=1e2, t_star=0.5, beta=10)
    assert isinstance(score.sde, TSVESDE)
    assert score.sde.sigma_min == 1e-3
    assert score.sde.sigma_max == 1e2
    assert score.sde.epsilon == 0
    assert score.sde.T == 1
    assert score.sde.t_star == 0.5
    assert score.sde.beta == 10


@pytest.mark.parametrize("method", ["Euler", "Heun", "RK4"])
@pytest.mark.parametrize("steps", [10, 20])
@pytest.mark.parametrize("cotangent_vectors", [1, 5, 10])
def test_log_prob(method, steps, cotangent_vectors):
    net = MLP(2)
    score = ScoreModel(net, sigma_min=1e-2, sigma_max=10)
    x = torch.randn(3, 2)
    ll = score.log_prob(x, steps=steps, method=method, cotangent_vectors=cotangent_vectors)
    print(ll)
    assert ll.shape == torch.Size([3])


if __name__ == "__main__":
    local_test_loading_model_and_score_fn()
