# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Improved diffusion model architecture proposed in the paper
Analyzing and Improving the Training Dynamics of Diffusion Models.

Original implementation by: Tero Karras (https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py)
Modification by: Alexandre Adam
"""
from typing import List, Optional, Literal
from torch import Tensor
from .conditional_branch_mp import (
    validate_conditional_arguments,
    conditional_branch,
    merge_conditional_time_branch,
    merge_conditional_input_branch,
)
import numpy as np
import torch
from ..layers import (
        mp_silu,
        mp_sum,
        mp_cat,
        MPConv, # Conv layer or Linear layer, depending on the input shape
        MPFourier,
        Block
        )


__all__ = ["EDMv2Net"]


#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(torch.nn.Module):
    def __init__(
        self,
        pixels: int,                              # Image resolution.
        channels: int,                            # Image channels.
        nf: int                         = 192,    # Base multiplier for the number of channels.
        ch_mult: List[int]              = [1,2,3,4], # Per-resolution multipliers for the number of channels.
        num_blocks: int                 = 3,      # Number of residual blocks per resolution.
        attn_resolutions: List[int]     = [16,8], # List of resolutions with self-attention.
        label_balance: float            = 0.5,    # Balance between noise embedding (0) and class embedding (1).
        concat_balance: float           = 0.5,    # Balance between skip connections (0) and main path (1).
        fourier_scale: float            = 0.02,   # Leads to much smoother score function (Lu & Song, https://arxiv.org/abs/2410.11081)
        conditions: Optional[Literal["discrete", "continuous", "vector", "tensor"]] = None,
        condition_channels: Optional[tuple[int]] = None,
        condition_embeddings: Optional[tuple[int]] = None,
        condition_balance: float        = 0.5,
        **block_kwargs,
    ):
        super().__init__()
        validate_conditional_arguments(conditions, condition_embeddings, condition_channels)
        self.conditioned = conditions is not None
        self.condition_type = conditions
        self.condition_embeddings = condition_embeddings
        self.condition_channels = condition_channels
        self.condition_balance = condition_balance
        default_block_kwargs = { # Global hyperparameter, we skip the layer specific ones
                "resample_filter": Block.__init__.__defaults__[2],
                "channels_per_head": Block.__init__.__defaults__[4],
                "dropout": Block.__init__.__defaults__[5],
                "res_balance": Block.__init__.__defaults__[6],
                "attn_balance": Block.__init__.__defaults__[7],
                "clip_act": Block.__init__.__defaults__[8],
                }
        default_block_kwargs.update(block_kwargs)
        self.hyperparameters = {
            "pixels": pixels,
            "channels": channels,
            "nf": nf,
            "ch_mult": ch_mult,
            "num_blocks": num_blocks,
            "attn_resolutions": attn_resolutions,
            "label_balance": label_balance,
            "concat_balance": concat_balance,
            "fourier_scale": fourier_scale,
            "conditions": conditions,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings,
            "condition_balance": condition_balance,
            **default_block_kwargs
        }

        cblock = [nf * m for m in ch_mult]
        cnoise = cblock[0]
        cemb = max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Conditional Branch
        if self.conditioned:
            total_time_channels, total_input_channels = conditional_branch(
                self,
                time_branch_channels=cnoise,
                input_branch_channels=channels,
                condition_embeddings=condition_embeddings,
                condition_channels=condition_channels,
                fourier_scale=fourier_scale,
            )  # This method attach a Module list to self.conditional_branch
        else:
            total_time_channels = cnoise
            total_input_channels = channels

        # Time Branch
        self.emb_fourier = MPFourier(cnoise, bandwidth=fourier_scale)
        self.emb_noise = MPConv(total_time_channels, cemb, kernel=[]) # Linear layer

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = total_input_channels + 1 # +1 for the bias channel
        for level, ch in enumerate(cblock):
            res = pixels >> level
            if level == 0:
                cin = cout
                cout = ch
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = ch
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, ch in reversed(list(enumerate(cblock))):
            res = pixels >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = ch
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, channels, kernel=[3,3])

    def forward(self, t, x, *args, **kwargs):
        # Embedding.
        emb = self.emb_fourier(t)
        if self.conditioned:
            emb = merge_conditional_time_branch(self, emb, *args, condition_balance=self.condition_balance)
        emb = self.emb_noise(emb)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        if self.conditioned:
            x = merge_conditional_input_branch(self, x, *args, condition_balance=self.condition_balance)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

class EDMv2Net(torch.nn.Module):
    """
    As described in Karras et al. 2023, the EDMv2Net is composed of the U-net 
    and an uncertainty estimation module to allow for the dynamic reweighting 
    of the noise levels during training.
    """
    def __init__(
        self,
        pixels: int,                 # Image resolution.
        channels: int,               # Image channels.
        logvar_channels: int = 128,  # Intermediate dimensionality for uncertainty estimation.
        **kwargs,               # Keyword arguments for UNet.
    ):
        super().__init__()
        self.pixels = pixels
        self.channels = channels
        self.unet = UNet(pixels=pixels, channels=channels, **kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        
        kwargs.update(self.unet.hyperparameters)
        self.hyperparameters = {
            "pixels": pixels,
            "channels": channels,
            "logvar_channels": logvar_channels,
            **kwargs
        }

    # kwargs need to be return_logvar, assumed to be the case in dsm.py
    def forward(self, t, x, *args, return_logvar=False, **kwargs):
        B, *D = x.shape
        out = self.unet(t, x, *args, **kwargs)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(t)).reshape(-1, *[1]*len(D))
            return out, logvar # u(sigma) in Equation 21
        return out

#----------------------------------------------------------------------------
