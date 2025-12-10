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

import torch
import numpy as np
from torch.func import vmap
from .utils import normalize
from .edm_resample import resample

left_matmul = vmap(torch.matmul, in_dims=(None, 0))

__all__ = ['mp_silu', 'mp_sum', 'mp_cat', 'MPFourier', 'MPConv', 'Block', 'MPPositionalEncoding']


#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):
    def __init__(self, width, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(width) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(width))

    def forward(self, t): 
        emb = t.to(torch.float32)
        emb = torch.outer(emb, self.freqs.to(torch.float32))
        emb = emb + self.phases.to(torch.float32)
        emb = emb.cos() * np.sqrt(2)
        return emb.to(t.dtype)

class MPPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, width, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(width, channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(width))

    def forward(self, x): 
        emb = x.to(torch.float32)
        emb = left_matmul(self.freqs, emb)
        emb = emb + self.phases.to(torch.float32)
        emb = emb.cos() * np.sqrt(2)
        return emb.to(t.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,                               # Number of input channels.
        out_channels: int,                              # Number of output channels.
        emb_channels: int,                              # Number of embedding channels.
        flavor: Literal["enc", "dec"]                = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode: Literal["keep", "up", "down"] = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter: List[int]                   = [1,1],    # Resampling filter.
        attention: bool                              = False,    # Include self-attention?
        channels_per_head: int                       = 64,       # Number of channels per attention head.
        dropout: float                               = 0,        # Dropout probability.
        res_balance: float                           = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float                          = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: Optional[float]                    = 256,      # Clip output activations. None = do not clip. (Karras et al. 2023 used clip_act=256)
        **kwargs, 
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x
