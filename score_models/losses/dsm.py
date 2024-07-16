from typing import Union
from torch import Tensor
import torch

__all__ = ["denoising_score_matching_loss"]

def denoising_score_matching_loss(model: Union["ScoreModel", "EnergyModel"], samples: Tensor, *args: list[Tensor]):
    B, *D = samples.shape
    sde = model.sde
    
    x0 = samples                                                                   # x0 ~ p(x0)
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(samples)                                                  # z ~ N(0, 1)
    
    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * samples + sigma * z                                                  # xt ~ p(xt | x0)
    
    # Compute the loss
    epsilon_theta = score_model.reparametrized_forward(t, xt, *args) # Numerically stable reparametrization for DSM loss
    return torch.sum((epsilon_theta + z)**2) / (2 * B)

# def denoising_score_matching_second_order_loss(
