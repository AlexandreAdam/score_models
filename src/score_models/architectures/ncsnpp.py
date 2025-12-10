from typing import Optional, Literal

import torch.nn as nn
import numpy as np
import torch
from functools import partial

from ..layers import (
    DDPMResnetBlock,
    ResnetBlockBigGANpp,
    GaussianFourierProjection,
    SelfAttentionBlock,
    UpsampleLayer,
    DownsampleLayer,
    Combine,
    conv3x3,
)
from ..utils import get_activation
from ..definitions import default_init
from .conditional_branch import (
    validate_conditional_arguments,
    conditional_branch,
    merge_conditional_time_branch,
    merge_conditional_input_branch,
)

__all__ = ["NCSNpp"]


class NCSNpp(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimensions: Literal[1, 2, 3] = 2,
        nf: int = 128,
        ch_mult: tuple[int] = (2, 2, 2, 2),
        num_blocks: int = 2,
        activation_type: str = "swish",
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        fir: bool = True,
        fir_kernel: tuple[int] = (1, 3, 3, 1),
        skip_rescale: bool = True,
        progressive: Literal["none", "output_skip", "residual"] = "output_skip",
        progressive_input: Literal["none", "input_skip", "residual"] = "input_skip",
        init_scale: float = 1e-2,
        fourier_scale: float = 0.02, # Leads to much smoother score function (Lu & Song, https://arxiv.org/abs/2410.11081)
        resblock_type: Literal["biggan", "ddpm"] = "biggan",
        combine_method: Literal["cat", "sum"] = "cat",
        attention: bool = True,
        conditions: Optional[
            tuple[Literal["time_discrete", "time_continuous", "time_vector", "input_tensor"]] # TODO: make better names
        ] = None,
        condition_embeddings: Optional[tuple[int]] = None,
        condition_channels: Optional[int] = None,
        # fourier_features=False,
        # n_min=7,
        # n_max=8,
        **kwargs,
    ):
        """
        NCSN++ model

        Args:
            channels (int): Number of input channels. Default is 1.
            dimensions (Literal[1, 2, 3]): Number of dimensions for input data. Default is 2.
            nf (int): Number of filters in the first layer. Default is 128.
            ch_mult (tuple[int]): Channel multiplier for each layer. Default is (2, 2, 2, 2).
            num_blocks (int): Number of residual blocks. Default is 2.
            activation_type (str): Type of activation function to use. Default is "swish".
            dropout (float): Dropout probability. Default is 0.
            resample_with_conv (bool): Whether to resample with convolution. Default is True.
            fir (bool): Whether to use finite impulse response filter. Default is True.
            fir_kernel (tuple[int]): Kernel size for FIR filter. Default is (1, 3, 3, 1).
            skip_rescale (bool): Whether to skip rescaling. Default is True.
            progressive (Literal["none", "output_skip", "residual"]): Type of progressive training. Default is "output_skip".
            progressive_input (Literal["none", "input_skip", "residual"]): Type of progressive input. Default is "input_skip".
            init_scale (float): Initial scale for weights. Default is 1e-2.
            fourier_scale (float): Scale for Fourier features. Default is 16.
            resblock_type (Literal["biggan", "ddpm"]): Type of residual block. Default is "biggan".
            combine_method (Literal["concat", "sum"]): Method for combining features. Default is "sum".
            attention (bool): Whether to use attention mechanism. Default is True.
            conditions (Optional[tuple[Literal["time_discrete", "time_continuous", "time_vector", "input_tensor"]]]): Conditions for input data. Default is None.
            condition_embeddings (Optional[tuple[int]]): Embedding size for conditions. Default is None.
            condition_channels (Optional[int]): Number of channels for conditions. Default is None.
        """
        super().__init__()
        # Backward compatibility
        num_blocks = kwargs.get("num_res_blocks", num_blocks)

        if dimensions not in [1, 2, 3]:
            raise ValueError(
                "Input must have 1, 2, or 3 spatial dimensions to use this architecture"
            )
        self.dimensions = dimensions
        validate_conditional_arguments(conditions, condition_embeddings, condition_channels)
        self.conditioned = conditions is not None
        self.condition_type = conditions
        self.condition_embeddings = condition_embeddings
        self.condition_channels = condition_channels
        self.channels = channels
        self.act = act = get_activation(activation_type)
        self.attention = attention
        self.nf = nf
        self.num_blocks = num_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.skip_rescale = skip_rescale
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.resblock_type = resblock_type
        if progressive not in ["none", "output_skip", "residual"]:
            raise ValueError(
                f"progressive must be in ['none', 'output_skip', 'residual'], received {progressive}"
            )
        if progressive_input not in ["none", "input_skip", "residual"]:
            raise ValueError(
                f"progressive_input must be in ['none', 'input_skip', 'residual'], received {progressive_input}"
            )
        self.hyperparameters = {
            "channels": channels,
            "nf": nf,
            "activation_type": activation_type,
            "ch_mult": ch_mult,
            "num_blocks": num_blocks,
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
            "dimensions": dimensions,
            "conditions": conditions,
            "condition_embeddings": condition_embeddings,
            "condition_channels": condition_channels,
            # "fourier_features": fourier_features,
            # "n_min": n_min,
            # "n_max": n_max
        }

        ########### Conditional branch ###########
        if self.conditioned:
            total_time_channels, total_input_channels = conditional_branch(
                self,
                time_branch_channels=nf,
                input_branch_channels=channels,
                condition_embeddings=condition_embeddings,
                condition_channels=condition_channels,
                fourier_scale=fourier_scale,
            )  # This method attach a Module list to self.conditional_branch
        else:
            total_time_channels = nf
            total_input_channels = channels
        #########################################

        ########### Time branch ###########
        modules = [
            GaussianFourierProjection(embed_dim=nf, scale=fourier_scale),  # Time embedding
            nn.Linear(
                total_time_channels, nf * 4
            ),  # Combine time embedding with conditionals if any
            nn.Linear(nf * 4, nf * 4),
        ]
        with torch.no_grad():
            modules[1].weight.data = default_init()(modules[1].weight.shape)
            modules[1].bias.zero_()
            modules[2].weight.data = default_init()(modules[2].weight.shape)
            modules[2].bias.zero_()
        ####################################

        ########### Prepare layers ###########
        combiner = partial(Combine, method=combine_method.lower(), dimensions=self.dimensions)
        AttnBlock = partial(SelfAttentionBlock, init_scale=init_scale, dimensions=dimensions)
        Upsample = partial(
            UpsampleLayer,
            with_conv=resample_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
            dimensions=self.dimensions,
        )
        Downsample = partial(
            DownsampleLayer,
            with_conv=resample_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
            dimensions=self.dimensions,
        )
        if progressive == "output_skip":
            self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == "residual":
            pyramid_upsample = partial(
                UpsampleLayer,
                fir=fir,
                fir_kernel=fir_kernel,
                with_conv=True,
                dimensions=self.dimensions,
            )
        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = partial(Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)
        if resblock_type == "ddpm":
            ResnetBlock = partial(
                DDPMResnetBlock,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                dimensions=self.dimensions,
            )

        elif resblock_type == "biggan":
            ResnetBlock = partial(
                ResnetBlockBigGANpp,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                dimensions=self.dimensions,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")
        #####################################

        # Downsampling block
        input_pyramid_ch = total_input_channels
        modules.append(conv3x3(total_input_channels, nf, dimensions=dimensions))
        hs_c = [nf]
        in_ch = nf  # + fourier_feature_channels
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(in_ch=input_pyramid_ch, out_ch=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
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
            for i_block in range(num_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=max(min(in_ch // 4, 32), 1), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, init_scale=init_scale, dimensions=dimensions)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=max(min(in_ch // 4, 32), 1), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True, dimensions=dimensions))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=max(min(in_ch // 4, 32), 1), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(
                            conv3x3(
                                in_ch,
                                channels,
                                bias=True,
                                init_scale=init_scale,
                                dimensions=dimensions,
                            )
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(num_groups=max(min(in_ch // 4, 32), 1), num_channels=in_ch, eps=1e-6)
            )
            modules.append(conv3x3(in_ch, channels, init_scale=1.0, dimensions=dimensions))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, x, *args, **kwargs):
        B, *D = x.shape
        modules = self.all_modules
        m_idx = 0

        # Time branch
        temb = modules[m_idx](t).view(B, -1)
        m_idx += 1
        if self.conditioned:
            temb = merge_conditional_time_branch(self, temb, *args)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Input branch
        if self.conditioned:
            x = merge_conditional_input_branch(self, x, *args)
        # if self.fourier_features:
        # ffeatures = self.fourier_features(x)
        # x = torch.concat([x, ffeatures], axis=1)
        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_blocks):
                h = modules[m_idx](hs[-1], temb)
                torch.var(h)
                m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
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
            for i_block in range(self.num_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")
            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1
        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        assert m_idx == len(modules)

        return h
