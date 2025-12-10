from typing import Optional, Literal

import torch
from torch import nn
from .mlp import MLP
from ..layers import GaussianFourierProjection


__all__ = ["MLPv2"]


class MLPv2(nn.Module):
    """
    Improved MLP architecture, with U-net skip connections and 
    magnitude preserving layers. It also incorporates the uncertainty 
    estimation layer from Karras et al. (2024).
    
    TODO: Incorporate the Magnitude preserving layer

    For now, we use MLPv1 and add the uncertainty estimation layer.
    """
    def __init__(
        self,
        channels: int,
        width: int = 100,
        layers: int = 4,
        fourier_scale: int = 0.02, # Leads to smoother score functions (Lu & Song, https://arxiv.org/abs/2410.11081)
        activation: int = "silu",
        attention: bool = False,
        nn_is_energy: bool = False,
        output_activation: str = None,
        conditions: Optional[Literal["discrete", "continuous", "vector", "tensor"]] = None,
        condition_channels: Optional[tuple[int]] = None,
        condition_embeddings: Optional[tuple[int]] = None,
        **kwargs,
        ):
        super().__init__()
        self.net = MLP(
            channels, 
            width, 
            layers, 
            fourier_scale, 
            activation, 
            attention, 
            nn_is_energy, 
            output_activation, 
            conditions, 
            condition_channels, 
            condition_embeddings, 
            **kwargs
        )
        self.logvar_fourier = GaussianFourierProjection(width, scale=fourier_scale)
        self.logvar_linear = nn.Linear(width, 1)
        self.hyperparameters = self.net.hyperparameters

    # kwargs need to be return_logvar, assumed to be the case in dsm.py
    def forward(self, t, x, *args, return_logvar=False, **kwargs):
        B, *D = x.shape
        out = self.net(t, x, *args, **kwargs)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(t)).reshape(-1, *[1]*len(D))
            return out, logvar # u(sigma) in Equation 21
        return out
