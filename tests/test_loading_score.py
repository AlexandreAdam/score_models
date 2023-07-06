import torch
from score_models.utils import load_architecture
from score_models import ScoreModel, EnergyModel
from score_models.architectures import MLP
import pytest


def local_test_loading_model():
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

def test_init_score():
    net = MLP(10)
    with pytest.raises(KeyError):
        score = ScoreModel(net)
    

    

    
