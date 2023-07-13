"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/
with slight modifications to make it work on continuous time.
"""
import torch
from torch import nn
from score_models.utils import get_activation
from score_models.layers import DDPMResnetBlock, SelfAttentionBlock, GaussianFourierProjection, UpsampleLayer, DownsampleLayer
from score_models.layers.ddpm_resnet_block import conv3x3
import functools


class DDPM(nn.Module):
    def __init__(
            self,
            channels=1,
            dimensions=2,
            nf=128,
            activation_type="relu",
            ch_mult=(1, 1, 2, 2, 4, 4),
            num_res_blocks=2,
            resample_with_conv=True,
            dropout=0.,
            attention=True,
            **kwargs
    ):
        super().__init__()
        self.hyperparameters = {
            "channels": channels,
            "nf": nf,
            "activation_type": activation_type,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "resample_with_conv": resample_with_conv,
            "dropout": dropout,
            "attention": attention,
            "dimensions": dimensions
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

        # Condition on continuous time
        modules = [GaussianFourierProjection(embed_dim=nf), nn.Linear(nf, nf * 4), nn.Linear(nf * 4, nf * 4)]
        with torch.no_grad():
            modules[1].bias.zero_()
            modules[2].bias.zero_()

        # Downsampling block
        Downsample = functools.partial(DownsampleLayer, dimensions=dimensions)
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
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
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=min(in_ch // 4, 32), eps=1e-6))
        modules.append(conv3x3(in_ch, channels))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, x):
        modules = self.all_modules
        m_idx = 0
        temb = t
        for _ in range(3):
            temb = modules[m_idx](temb)
            m_idx += 1

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

