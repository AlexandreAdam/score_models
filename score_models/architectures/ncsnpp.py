from score_models.layers import DDPMResnetBlock, GaussianFourierProjection, SelfAttentionBlock, \
        UpsampleLayer, DownsampleLayer, Combine, ResnetBlockBigGANpp, conv3x3
from score_models.utils import get_activation
from score_models.definitions import default_init
import torch.nn as nn
import functools
import torch
import numpy as np


class NCSNpp(nn.Module):
    """
    NCSN++ model

    Args:
        channels (int): Number of input channels. Default is 1.
        dimensions (int): Number of dimensions of the input data. Default is 2.
        nf (int): Number of filters in the first layer. Default is 128.
        ch_mult (tuple): Channel multiplier for each layer. Default is (2, 2, 2, 2).
        num_res_blocks (int): Number of residual blocks in each layer. Default is 2.
        activation_type (str): Type of activation function. Default is "swish".
        dropout (float): Dropout probability. Default is 0.
        resample_with_conv (bool): Whether to use convolutional resampling. Default is True.
        fir (bool): Whether to use finite impulse response filtering. Default is True.
        fir_kernel (tuple): FIR filter kernel. Default is (1, 3, 3, 1).
        skip_rescale (bool): Whether to rescale skip connections. Default is True.
        progressive (str): Type of progressive training. Default is "output_skip".
        progressive_input (str): Type of progressive
        init_scale (float): The initial scale for the function. Default is 1e-2.
        fourier_scale (float): The Fourier scale for the function. Default is 16.
        resblock_type (str): The type of residual block to use. Default is "biggan".
        combine_method (str): The method to use for combining the results. Default is "sum".
        attention (bool): Whether or not to use attention. Default is True.

    """
    def __init__(
            self,
            channels=1,
            dimensions=2,
            nf=128,
            ch_mult=(2, 2, 2, 2),
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
            attention=True,
            **kwargs
          ):
        super().__init__()
        assert dimensions in [1, 2, 3]
        self.dimensions = dimensions
        self.channels = channels
        self.hyperparameters = {
            "channels": channels,
            "nf": nf,
            "activation_type": activation_type,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "resample_with_conv": resample_with_conv,
            "dropout": dropout,
            "fir": fir,
            "fir_kernel": fir_kernel,
            "skip_rescale": skip_rescale,
            "progressive": progressive,
            "progressive_input": progressive_input,
            "init_scale": init_scale,
            "fourier_scale": fourier_scale,
            "resblock_type": resblock_type,
            "combine_method": combine_method,
            "attention": attention,
            "dimensions": dimensions
        }
        self.act = act = get_activation(activation_type)
        self.attention = attention

        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)

        self.skip_rescale = skip_rescale
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.resblock_type = resblock_type
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        combiner = functools.partial(Combine, method=combine_method.lower(), dimensions=self.dimensions)

        # Condition on continuous time
        modules = [GaussianFourierProjection(embed_dim=nf, scale=fourier_scale), nn.Linear(nf, nf * 4), nn.Linear(nf * 4, nf * 4)]
        with torch.no_grad():
            modules[1].weight.data = default_init()(modules[1].weight.shape)
            modules[1].bias.zero_()
            modules[2].weight.data = default_init()(modules[2].weight.shape)
            modules[2].bias.zero_()

        AttnBlock = functools.partial(SelfAttentionBlock, init_scale=init_scale, dimensions=dimensions)
        Upsample = functools.partial(UpsampleLayer, with_conv=resample_with_conv, fir=fir, fir_kernel=fir_kernel, dimensions=self.dimensions)

        if progressive == 'output_skip':
            self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(UpsampleLayer, fir=fir, fir_kernel=fir_kernel, with_conv=True, dimensions=self.dimensions)

        Downsample = functools.partial(DownsampleLayer, with_conv=resample_with_conv, fir=fir, fir_kernel=fir_kernel, dimensions=self.dimensions)

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
                                            temb_dim=nf * 4,
                                            dimensions=self.dimensions
                                            )

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            dimensions=self.dimensions
                                            )

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf, dimensions=dimensions))
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
                    modules.append(combiner(in_ch=input_pyramid_ch, out_ch=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.attention:
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
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale, dimensions=dimensions))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True, dimensions=dimensions))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale, dimensions=dimensions))
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
            modules.append(conv3x3(in_ch, channels, init_scale=1., dimensions=dimensions))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, x):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        # Gaussian Fourier features embeddings.
        temb = modules[m_idx](t)
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
        if self.attention:
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

