from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from score_models import ScoreModel, HessianDiagonal

from torch import Tensor
import torch

__all__ = [
        "dsm", 
        "denoising_score_matching", 
        "second_order_dsm", 
        "second_order_dsm_meng_variation"
        ]


def dsm(model: "ScoreModel", samples: Tensor, *args: list[Tensor]):
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
    epsilon_theta = model.reparametrized_score(t, xt, *args)                       # epsilon_theta(t, x) = sigma(t) * s(t, x)
    return ((epsilon_theta + z)**2).sum() / (2 * B)


def denoising_score_matching(model: "ScoreModel", samples: Tensor, *args: list[Tensor]):
    # Used for backward compatibility
    return dsm(model, samples, *args)


def second_order_dsm(model: "HessianDiagonal", samples: Tensor, *args: list[Tensor]):
    """
    Loss used to train a model to approximate the diagonal of the Hessians of log p(x).
    This loss is derived in the works of Meng et al. (2020), arxiv.org/pdf/2111.04726
    and Lu et al (2022), arxiv.org/pdf/2206.08265.
    
    In particular, this loss corresponds to equation (13) of Lu et al. (2022). It can be viewed 
    as a continuous time extension of equation (11) of Meng et al. (2020).
    
    A better version of this loss is implemented below in the spirit of equation (17) of 
    Meng et al. (2020).
    """
    B, *D = samples.shape
    sde = model.sde
    epsilon_model = model.score_model.reparametrized_score                         # epsilon_1(t, x) = sigma(t) * s_1(t, x)
    
    # Compute the first order DSM loss
    x0 = samples                                                                   # x0 ~ p(x0)
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(samples)                                                  # z ~ N(0, 1)

    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * samples + sigma * z                                                  # xt ~ p(xt | x0)
    with torch.no_grad():
        ell_1 = epsilon_model(t, xt, *args) + z                                    # ell_1 is the DSM loss term before contraction
    
    # Compute the second order DSM loss
    diag_theta = model.reparametrized_diagonal(t, xt, *args)                       # diag_theta(t, x) = sigma(t)**2 * diag(s_2(t, x)) + 1
    return ((diag_theta - ell_1**2)**2).sum() / (2 * B)

def second_order_dsm_meng_variation(model: "HessianDiagonal", samples: Tensor, *args: list[Tensor]):
    """
    Loss used to train a model to approximate the diagonal of the Hessians of log p(x).
    This loss is derived in the works of Meng et al. (2020), arxiv.org/pdf/2111.04726
    and Lu et al (2022), arxiv.org/pdf/2206.08265.

    This loss corresponds to equation (17) of Meng et al. (2020) extended to continuous time  
    as a more stable version of the loss in second_order_noisy_dsm.
    """
    B, *D = samples.shape
    sde = model.sde
    epsilon_model = model.score_model.reparametrized_score                         # epsilon_1(t, x) = sigma(t) * s_1(t, x)
    
    # Compute the first order DSM loss
    x0 = samples                                                                   # x0 ~ p(x0)
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(samples)                                                  # z ~ N(0, 1)

    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * samples + sigma * z                                                  # xt ~ p(xt | x0)
    with torch.no_grad():
        epsilon_1 = epsilon_model(t, xt, *args) 
    
    # Compute the second order DSM loss
    diag_theta = model.reparametrized_diagonal(t, xt, *args)                       # diag_theta(t, x) = sigma(t)**2 * diag(s_2(t, x)) + 1
    return ((diag_theta + epsilon_1**2 - z**2)**2).sum() / (2 * B)
