from typing import Union
import torch
from torch import Tensor

def denoising_score_matching(score_model: Union["ScoreModel", "EnergyModel"], samples: Tensor, *args: list[Tensor]):
    B, *D = samples.shape
    sde = score_model.sde
    z = torch.randn_like(samples)
    t = torch.rand(B).to(score_model.device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, samples)
    return torch.sum((z + score_model.model(t, mean + sigma * z, *args)) ** 2) / B

