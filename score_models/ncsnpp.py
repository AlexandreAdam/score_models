# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .layers import DDPMResnetBlock, GaussianFourierProjection, SelfAttentionBlock, Conv2dSame
from .layers.resnet_block_biggan import ResnetBlockBigGANpp, upsample_2d, Conv2d, downsample_2d
from .utils import get_activation
from score_models.definitions import default_init, DEVICE
from score_models.sde import VESDE
import torch.nn.functional as F
import torch.nn as nn
import functools
import torch
import numpy as np
from tqdm import tqdm


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.):
    """1x1 convolution with DDPM initialization."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
    with torch.no_grad():
        conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
        nn.init.zeros_(conv.conv.bias)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with DDPM initialization."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=bias)
    with torch.no_grad():
        conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
        nn.init.zeros_(conv.conv.bias)
    return conv


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(in_ch, out_ch,
                                        kernel=3, down=True,
                                        resample_kernel=fir_kernel,
                                        use_bias=True,
                                        kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(in_ch, out_ch,
                                   kernel=3, up=True,
                                   resample_kernel=fir_kernel,
                                   use_bias=True,
                                   kernel_init=default_init())
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)

        return h


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(
            self,
            channels,
            image_size=256,
            sigma_min=1e-1,
            sigma_max=50,
            nf=128,
            ch_mult=(1, 1, 2, 2, 2, 2, 2),
            num_res_blocks=2,
            activation_type="swish",
            dropout=0.,
            resample_with_conv=True,
            fir=True,
            fir_kernel=(1, 3, 3, 1),
            skip_rescale=True,
            progressive="output_skip",
            progressive_input="input_skip",
            init_scale=1e-2,
            fourier_scale=16.,
            resblock_type="biggan",
            combine_method="sum",
            **kwargs
          ):
        super().__init__()
        self.act = act = get_activation(activation_type)
        self.sde = VESDE(sigma_min, sigma_max)

        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.skip_rescale = skip_rescale
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.resblock_type = resblock_type
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        combiner = functools.partial(Combine, method=combine_method.lower())

        # Condition on continuous time
        modules = [GaussianFourierProjection(embed_dim=nf, scale=fourier_scale), nn.Linear(nf, nf * 4), nn.Linear(nf * 4, nf * 4)]
        with torch.no_grad():
            modules[1].weight.data = default_init()(modules[1].weight.shape)
            modules[1].bias.zero_()
            modules[2].weight.data = default_init()(modules[2].weight.shape)
            modules[2].bias.zero_()

        AttnBlock = functools.partial(SelfAttentionBlock, init_scale=init_scale)
        Upsample = functools.partial(UpsampleLayer, with_conv=resample_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(UpsampleLayer, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(DownsampleLayer, with_conv=resample_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(DDPMResnetBlock,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        input_pyramid_ch = channels

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
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=1.))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        # Gaussian Fourier features embeddings.
        temb = modules[m_idx](time_cond)
        m_idx += 1
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                torch.var(h)
                m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1
        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        assert m_idx == len(modules)

        return h

    def score(self, x, t):
        B, *D = x.shape
        return self.forward(x, t) / self.sde.sigma(t).view(B, *[1]*len(D))

    def sample(self, size, N: int = 1000, device=DEVICE):
        assert len(size) == 4
        assert size[1] == 1
        assert N > 0
        # A simple Euler-Maruyama integration of VESDE
        x = torch.randn(size).to(device)
        dt = -1.0 / N
        t = torch.ones(size[0]).to(DEVICE)
        broadcast = [-1, 1, 1, 1]
        for _ in tqdm(range(N)):
            t += dt
            drift, diffusion = self.sde.sde(x, t)
            score = self.score(x, t)
            drift = drift - diffusion.view(*broadcast)**2 * score
            z = torch.randn_like(x)
            x_mean = x + drift * dt
            x = x_mean + diffusion.view(*broadcast) * (-dt)**(1/2) * z
        return x_mean
