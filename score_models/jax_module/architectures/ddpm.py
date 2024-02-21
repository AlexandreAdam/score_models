"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/
with slight modifications to make it work on continuous time.
"""
from typing import Callable
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from ..utils import get_activation
from ..layers import (
    GaussianFourierProjection,
    DDPMResnetBlock,
    DownsampleLayer,
    UpsampleLayer,
    SelfAttentionBlock,
    conv3x3,
)


class DDPM(eqx.Module):
    modules: list
    act: Callable
    attention: bool
    num_resolutions: int

    def __init__(
        self,
        channels=1,
        dimensions=2,
        nf=128,
        activation_type="relu",
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        resample_with_conv=True,
        dropout=0.0,
        attention=True,
        conditioning=["None"],
        conditioning_channels=None,
    ):
        if dimensions not in [1, 2, 3]:
            raise ValueError(
                f"Input must have 1, 2, or 3 spatial dimensions, received {dimensions}."
            )

        self.act = get_activation(activation_type)
        self.attention = attention
        self.num_resolutions = len(ch_mult)

        AttnBlock = SelfAttentionBlock
        ResnetBlock = partial(
            DDPMResnetBlock,
            act=self.act,
            temb_dim=4 * nf,
            dropout=dropout,
            dimensions=dimensions,
        )

        modules = []
        modules += [
            GaussianFourierProjection(embed_dim=nf),
            eqx.nn.Linear(nf, nf * 4),
            eqx.nn.Linear(nf * 4, nf * 4),
        ]

        Downsample = partial(DownsampleLayer, dimensions=dimensions)
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            out_ch = nf * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                modules.append(Downsample(in_ch=in_ch, with_conv=resample_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.attention:
            modules.append(AttnBlock(in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        Upsample = partial(UpsampleLayer, dimensions=dimensions)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if i_level != 0:
                modules.append(Upsample(in_ch=in_ch, with_conv=resample_with_conv))

        modules.append(
            eqx.nn.GroupNorm(num_channels=in_ch, num_groups=min(in_ch // 4, 32))
        )
        modules.append(conv3x3(in_ch, channels))
        self.modules = modules

    def __call__(self, t, x):
        m_idx = 0
        temb = t
        for _ in range(3):
            temb = self.modules[m_idx](temb)
            m_idx += 1

        hs = [self.modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = self.modules[m_idx](h, temb)
        m_idx += 1
        if self.attention:
            h = self.modules[m_idx](h)
            m_idx += 1
        h = self.modules[m_idx](h, temb)
        m_idx += 1

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.modules[m_idx](jnp.concatenate([h, hs.pop()], axis=1), temb)
                m_idx += 1
            if i_level != 0:
                h = self.modules[m_idx](h)
                m_idx += 1

        h = self.act(self.modules[m_idx](h))
        m_idx += 1
        h = self.modules[m_idx](h)
        m_idx += 1
        return h
