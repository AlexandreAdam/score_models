import torch
from score_models.utils import load_architecture
from score_models import ScoreModel, EnergyModel


def test_loading_model():
    # local test only
    path = "/home/alexandre/Desktop/Projects/data/score_models/ncsnpp_ct_g_220912024942"
    model, hparams = load_architecture(path)
    
    score = ScoreModel(checkpoints_directory=path)
    print(score.sde)
    x = torch.randn(1, 1, 256, 256)
    t = torch.ones(1)
    score(t, x)
    
