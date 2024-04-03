"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/
with slight modifications to make it work on continuous time.
"""
import jax.numpy as jnp
from jax import random
import equinox as eqx
from functools import partial
from jaxtyping import Array, PRNGKeyArray
from ..utils import get_activation
from ..layers import (
    GaussianFourierProjection,
    DDPMResnetBlock,
    DownsampleLayer,
    UpsampleLayer,
    SelfAttentionBlock,
    conv3x3,
)


__all__ = ["DDPM"]


class DDPM(eqx.Module):
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
        *,
        key: PRNGKeyArray
    ):
        if dimensions not in [1, 2, 3]:
            raise ValueError(
                f"Input must have 1, 2, or 3 spatial dimensions, received {dimensions}."
            )

        self.activation = get_activation(activation_type)
        self.attention = attention
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        AttnBlock = SelfAttentionBlock
        ResnetBlock = partial(
            DDPMResnetBlock,
            act=self.activation,
            temb_dim=4 * nf,
            dropout=dropout,
            dimensions=dimensions,
        )

        modules = []
        key_time, key_rest = random.split(key)
        key_t1, key_t2, key_t3 = random.split(key_time, 3)
        modules += [
            GaussianFourierProjection(embed_dim=nf, key=key_t1),
            eqx.nn.Linear(nf, nf * 4, key=key_t2),
            eqx.nn.Linear(nf * 4, nf * 4, key=key_t3),
        ]

        Downsample = partial(DownsampleLayer, dimensions=dimensions)
        key_input, key_rest = random.split(key_rest)
        modules.append(conv3x3(channels, nf, key=key_input))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            out_ch = nf * ch_mult[i_level]
            key_layer, key_rest = random.split(key_rest)
            for i_block in range(num_res_blocks):
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, key=key_layer))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                key_downsample, key_rest = random.split(key_rest)
                modules.append(Downsample(in_ch=in_ch, with_conv=resample_with_conv, key=key_downsample))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        key_bottleneck, key_rest = random.split(key_rest)
        modules.append(ResnetBlock(in_ch=in_ch, key=key_bottleneck))
        if self.attention:
            key_attention, key_rest = random.split(key_rest)
            modules.append(AttnBlock(in_ch, key=key_attention))
        key_bottleneck2, key_rest = random.split(key_rest)
        modules.append(ResnetBlock(in_ch=in_ch, key=key_bottleneck2))

        Upsample = partial(UpsampleLayer, dimensions=dimensions)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                key_resnet, key_rest = random.split(key_rest)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch, key=key_resnet))
                in_ch = out_ch
            if i_level != 0:
                key_upsample, key_rest = random.split(key_rest)
                modules.append(Upsample(in_ch=in_ch, with_conv=resample_with_conv, key=key_upsample))

        modules.append(eqx.nn.GroupNorm(channels=in_ch, groups=min(in_ch // 4, 32)))
        key_output, key_rest = random.split(key_rest)
        modules.append(conv3x3(in_ch, channels, key=key_output))
        self.modules = modules

    def __call__(self, t: Array, x: Array) -> Array:
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

        h = self.activation(self.modules[m_idx](h))
        m_idx += 1
        h = self.modules[m_idx](h)
        m_idx += 1
        return h
