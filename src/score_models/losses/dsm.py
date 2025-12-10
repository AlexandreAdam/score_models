from typing import Union, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from score_models import ScoreModel, HessianDiagonal

from torch import Tensor
from contextlib import nullcontext
import torch

__all__ = [
        "dsm",
        "karras_dsm",
        "denoising_score_matching", 
        "second_order_dsm", 
        "second_order_dsm_meng_variation",
        "joint_second_order_dsm"
        ]


def dsm(model: "ScoreModel", x: Tensor, *args: list[Tensor], **kwargs):
    """
    Original preconditioning used by Yang Song and Jonathan Ho.
    """
    B, *D = x.shape
    sde = model.sde
    
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(x)                                                        # z ~ N(0, 1)
    
    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * x + sigma * z                                                        # xt ~ p(xt | x0)
    
    # Compute the loss
    epsilon_theta = model.reparametrized_score(t, xt, *args)                       # epsilon_theta(t, x) = sigma(t) * score(t, x)
    return ((epsilon_theta + z)**2).sum() / (2 * B)

def karras_dsm(model: "ScoreModel", x: Tensor, *args: list[Tensor], adaptive_loss: bool = False, **kwargs):
    """
    Desnoing Score Matching loss used by Tero Karras in his EDM formulation. 
    The idea is to use the Tweedie formula to train the score, and define 
    skip connection to stabilize the training at all temperatures.
    
    We also make use of a user defined sampling distribution for the time 
    index. This is used to improve significanlty the convergence of the model,
    by sampling more example in the spin-glass phase transition of the distribution.
    """
    B, *D = x.shape
    sde = model.sde
    
    t = model.sample_noise_level(B)
    z = torch.randn_like(x)                                                        # z ~ N(0, 1)
    
    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * x + sigma * z                                                        # xt ~ p(xt | x0)
    
    # EDM Denoising loss, with weight factor taken into account
    if adaptive_loss:
        F_theta, u = model.reparametrized_score(t, xt, *args, return_logvar=True)
        u = u.view(-1, *[1]*len(D))
    else:
        F_theta = model.reparametrized_score(t, xt, *args)
        u = torch.zeros_like(t).view(-1, *[1]*len(D))
    c_out = model.sde.c_out(t).view(B, *[1]*len(D))
    c_skip = model.sde.c_skip(t).view(B, *[1]*len(D))
    effective_score = (x - c_skip * xt) / c_out
    loss = (F_theta - effective_score)**2
    return (torch.exp(-u) * loss + u).sum() / (2 * B)

def denoising_score_matching(
        model: "ScoreModel", 
        x: Tensor, 
        *args: list[Tensor], 
        formulation: Literal["original", "edm"] = "edm", 
        **kwargs):
    if formulation == "original":
        return dsm(model, x, *args)
    elif formulation == "edm":
        return edm_dsm(model, x, *args)

# Note Meng's version is completely equivalent to Lu's version coded here
def second_order_dsm(model: "HessianDiagonal", x: Tensor, *args: list[Tensor], no_grad: bool = True, **kwargs):
    """
    Loss used to train a model to approximate the diagonal of the Hessians of log p(x).
    This loss is derived in the works of Meng et al. (2020), arxiv.org/pdf/2111.04726
    and Lu et al (2022), arxiv.org/pdf/2206.08265.
    
    In particular, this loss corresponds to equation (13) of Lu et al. (2022). It can be viewed 
    as a continuous time extension of equation (11) of Meng et al. (2020).
    
    A better version of this loss is implemented below in the spirit of equation (17) of 
    Meng et al. (2020).
    """
    B, *D = x.shape
    sde = model.sde
    epsilon_model = model.score_model.reparametrized_score                         # epsilon_1(t, x) = sigma(t) * s_1(t, x)
    
    # Compute the first order DSM loss
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(x)                                                  # z ~ N(0, 1)

    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * x + sigma * z                                                  # xt ~ p(xt | x0)
    with torch.no_grad() if no_grad else nullcontext():
        ell_1 = epsilon_model(t, xt, *args) + z                                    # ell_1 is the DSM loss term before contraction
    
    # Compute the second order DSM loss
    diag_theta = model.reparametrized_diagonal(t, xt, *args)                       # diag_theta(t, x) = sigma(t)**2 * diag(s_2(t, x)) + 1
    return ((diag_theta - ell_1**2)**2).sum() / (2 * B)

def second_order_dsm_meng_variation(model: "HessianDiagonal", x: Tensor, *args: list[Tensor], no_grad: bool = True, **kwargs):
    """
    Loss used to train a model to approximate the diagonal of the Hessians of log p(x).
    This loss is derived in the works of Meng et al. (2020), arxiv.org/pdf/2111.04726
    and Lu et al (2022), arxiv.org/pdf/2206.08265.

    This loss corresponds to equation (17) of Meng et al. (2020) extended to continuous time  
    as a more stable version of the loss in second_order_noisy_dsm.
    """
    B, *D = x.shape
    sde = model.sde
    epsilon_model = model.score_model.reparametrized_score                         # epsilon_1(t, x) = sigma(t) * s_1(t, x)
    
    # Compute the first order DSM loss
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(x)                                                        # z ~ N(0, 1)

    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * x + sigma * z                                                        # xt ~ p(xt | x0)
    with torch.no_grad() if no_grad else nullcontext():
        epsilon_1 = epsilon_model(t, xt, *args) 
    
    # Compute the second order DSM loss
    diag_theta = model.reparametrized_diagonal(t, xt, *args)                       # diag_theta(t, x) = sigma(t)**2 * diag(s_2(t, x)) + 1
    return ((diag_theta + epsilon_1**2 - z**2)**2).sum() / (2 * B)

def joint_second_order_dsm(model: "HessianDiagonal", x: Tensor, *args: list[Tensor], lambda_1: float = 1., **kwargs):
    """
    Joint optimization of the first and second order DSM losses.
    """
    B, *D = x.shape
    sde = model.sde
    
    t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon       # t ~ U(epsilon, T)
    z = torch.randn_like(x)                                                        # z ~ N(0, 1)
    
    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    xt = mu * x + sigma * z                                                        # xt ~ p(xt | x0)
    
    # Compute the DSM loss for the SBM
    epsilon_theta = model.score_model.reparametrized_score(t, xt, *args)           # epsilon_theta(t, x) = sigma(t) * s(t, x)
    ell_1 = epsilon_theta + z                                                      # ell_1 is the DSM loss term before contraction
    dsm_loss = (ell_1**2).sum() / (2 * B)

    # Compute the second order DSM loss
    diag_theta = model.reparametrized_diagonal(t, xt, *args)                       # diag_theta(t, x) = sigma(t)**2 * diag(s_2(t, x)) + 1
    second_order_loss = ((diag_theta - ell_1**2)**2).sum() / (2 * B)
    return dsm_loss + lambda_1 * second_order_loss
