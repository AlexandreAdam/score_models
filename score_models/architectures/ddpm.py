"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/
with slight modifications to make it work on continuous time.
"""
from typing import Optional, Literal

import torch
from torch import nn
import functools

from .conditional_branch import validate_conditional_arguments, conditional_branch
from ..utils import get_activation
from ..layers import (
        DDPMResnetBlock, 
        SelfAttentionBlock, 
        GaussianFourierProjection, 
        UpsampleLayer, 
        DownsampleLayer,
        conv3x3
        )

__all__ = ["DDPM"]

class DDPM(nn.Module):
    def __init__(
            self,
            channels: int = 1,
            dimensions: int = 2,
            nf: int = 128,
            activation_type: str = "relu",
            ch_mult: tuple[int] = (2, 2),
            num_res_blocks: int = 2,
            resample_with_conv: bool = True,
            dropout: float = 0.,
            attention: bool = True,
            fourier_scale: float = 30.,
            conditions : Optional[tuple[Literal["time_discrete", "time_continuous", "time_vector", "input_tensor"]]] = None,
            condition_embeddings:  Optional[tuple[int]] = None,
            condition_channels: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        if dimensions not in [1, 2, 3]:
            raise ValueError(f"Input must have 1, 2, or 3 spatial dimensions to use this architecture, received {dimensions}.")
        validate_conditional_arguments(conditions, condition_embeddings, condition_channels)
        self.conditioned = conditions is not None
        self.condition_type = conditions
        self.condition_embeddings = condition_embeddings
        self.condition_channels = condition_channels
        self.hyperparameters = {
            "channels": channels,
            "dimensions": dimensions,
            "nf": nf,
            "activation_type": activation_type,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "resample_with_conv": resample_with_conv,
            "dropout": dropout,
            "attention": attention,
            "fourier_scale": fourier_scale,
            "conditions": conditions,
            "condition_embeddings": condition_embeddings,
            "condition_channels": condition_channels
        }
        self.dimensions = dimensions
        self.act = act = get_activation(activation_type=activation_type)
        self.attention = attention
        self.channels = channels
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)

        AttnBlock = SelfAttentionBlock
        ResnetBlock = functools.partial(DDPMResnetBlock, act=act, temb_dim=4 * nf, dropout=dropout, dimensions=dimensions)

        ########### Conditional branch ###########
        if self.conditioned:
            total_time_channels, total_input_channels = conditional_branch(
                    self,
                    time_branch_channels=nf,
                    input_branch_channels=channels,
                    condition_embeddings=condition_embeddings,
                    condition_channels=condition_channels,
                    fourier_scale=fourier_scale
                    ) # This method attach a Module list to self.conditional_branch
        else:
            total_time_channels = nf
            total_input_channels = channels
        #########################################

        ########### Time branch ###########
        modules = [
                GaussianFourierProjection(embed_dim=nf, scale=fourier_scale),
                nn.Linear(total_time_channels, nf * 4), 
                nn.Linear(nf * 4, nf * 4)
                ]
        with torch.no_grad():
            modules[1].bias.zero_()
            modules[2].bias.zero_()
        ####################################

        # Downsampling block
        Downsample = functools.partial(DownsampleLayer, dimensions=dimensions)
        modules.append(conv3x3(total_input_channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            out_ch = nf * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(in_ch=in_ch, with_conv=resample_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.attention:
            modules.append(AttnBlock(in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        Upsample = functools.partial(UpsampleLayer, dimensions=dimensions)
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if i_level != 0:
                modules.append(Upsample(in_ch=in_ch, with_conv=resample_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=max(min(in_ch // 4, 32), 1), eps=1e-6))
        modules.append(conv3x3(in_ch, channels))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, x, *args):
        B, *D = x.shape
        modules = self.all_modules
        m_idx = 0
        temb = modules[m_idx](t)
        m_idx += 1

        # Append conditional to time branch
        c_idx = 0
        if self.conditioned:
            if len(args) != len(self.condition_type):
                raise ValueError(f"The network requires {len(self.condition_type)} additional arguments, but {len(args)} were provided.")
            for j, condition in enumerate(args):
                if "time" in self.condition_type[j].lower():
                    c_emb = self.conditional_branch[c_idx](condition).view(B, -1)
                    temb = torch.cat([temb, c_emb], dim=1)
                    c_idx += 1
        
        # Time branch
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Add conditionals to input branch
        if self.conditioned:
            for j, condition in enumerate(args):
                if "input" in self.condition_type[j].lower():
                    x = torch.cat([x, condition], dim=1)
        
        # Add Fourier features to input branch
        # if self.fourier_features:
            # ffeatures = self.fourier_features(x)
            # x = torch.concat([x, ffeatures], axis=1)

        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        if self.attention:
            h = modules[m_idx](h)
            m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        return h
