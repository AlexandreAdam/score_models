from typing import Literal, Optional

import torch
from torch import nn
from .conditional_branch import (
    validate_conditional_arguments,
    conditional_branch,
    merge_conditional_time_branch,
    merge_conditional_input_branch,
)
from ..definitions import default_init
from ..layers import Conv2dSame, ResnetBlockBigGANpp, GaussianFourierProjection
from ..utils import get_activation


__all__ = ["Encoder"]


class Encoder(nn.Module):
    """
    Function that ouputs latent representations of an 1D, 2D or 3D random variable
    (i.e. shape = [D, C, *D] where D is the number of dimensions,
    C is the number of channels and *D are the spatial dimensions)
    conditioned on time and possitbly other variables.
    """

    def __init__(
        self,
        pixels: int,
        channels: int,
        latent_size: int,
        input_kernel_size=7,  # Kernel size of the first convolutional layer
        nf: int = 64,  # Base width of the convolutional layers
        ch_mult: tuple[int] = (2, 2, 2, 2),  # Channel multiplier for each level
        num_res_blocks: int = 2,
        activation: Literal["relu", "gelu", "leakyrelu", "sigmoid", "tanh", "silu"] = "silu",
        output_kernel: int = 2,  # Final layer is an average pooling layer with this kernel shape
        hidden_layers: int = 1,
        hidden_size: int = 256,
        factor: int = 2,
        fourier_scale: float = 0.02,
        conditions: Optional[
            tuple[Literal["time_discrete", "time_continuous", "time_vector", "input_tensor"]]
        ] = None,
        condition_embeddings: Optional[tuple[int]] = None,
        condition_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Function that ouputs latent representations of an 1D, 2D or 3D random variable of shape
            shape = [D, C, *D]
        where D is the number of dimensions, C is the number of channels and *D are the spatial dimensions).
        This network is conditioned on time and possitbly other variables.

        Parameters:
        - pixels (int): The number of pixels in the input image.
        - channels (int): The number of channels in the input image.
        - latent_size (int): The size of the latent representation.
        - input_kernel_size (int): Kernel size of the first convolutional layer (default is 7).
        - nf (int): Base width of the convolutional layers (default is 64).
        - ch_mult (tuple[int]): Channel multiplier for each level (default is (2, 2, 2, 2)).
        - num_res_blocks (int): Number of residual blocks (default is 2).
        - activation (str): Activation function to use (options: "relu", "gelu", "leakyrelu", "sigmoid", "tanh", "silu", default is "silu").
        - output_kernel (int): Kernel size of the final average pooling layer (default is 2).
        - hidden_layers (int): Number of hidden layers (default is 1).
        - hidden_size (int): Size of the hidden layers (default is 256).
        - factor (int): Factor to scale the hidden size by (default is 2).
        - conditions (tuple[str]): Types of conditions to consider (options: "time_discrete", "time_continuous", "time_vector", "input_tensor").
        - condition_embeddings (tuple[int]): Embedding sizes for the conditions.
        - condition_channels (int): Number of channels for the conditions.

        """
        super().__init__()
        validate_conditional_arguments(conditions, condition_embeddings, condition_channels)
        self.conditioned = conditions is not None
        self.hyperparameters = {
            "pixels": pixels,
            "channels": channels,
            "latent_size": latent_size,
            "nf": nf,
            "input_kernel_size": input_kernel_size,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "activation": activation,
            "hidden_layers": hidden_layers,
            "hidden_size": hidden_size,
            "output_kernel": output_kernel,
            "factor": factor,
            "fourier_scale": fourier_scale,
            "conditions": conditions,
            "condition_embeddings": condition_embeddings,
            "condition_channels": condition_channels,
        }
        assert (output_kernel % 2 == 0) or (
            output_kernel == 1
        ), "output_kernel must be an even number or equal to 1 (no average pooling at the end)"
        assert pixels % 2 ** len(ch_mult) == 0, "pixels must be divisible by 2**len(ch_mult)"

        self.act = get_activation(activation)
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.pixels = pixels
        self.channels = channels
        self.factor = factor
        self._latent_pixels = pixels // factor ** (len(ch_mult) + output_kernel // 2)
        self._latent_channels = int(nf * ch_mult[-1])
        assert (
            self._latent_pixels > 0
        ), "Network is too deep for the given input size and downsampling factor"

        ### Conditional branch ###
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

        ### Time branch ###
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
        self.time_branch = nn.ModuleList(modules)

        ### Input branch ###
        out_ch = int(nf * ch_mult[0])
        self.input_layer = Conv2dSame(total_input_channels, out_ch, kernel_size=input_kernel_size)
        layers = []
        for i in range(len(ch_mult)):
            in_ch = out_ch = int(nf * ch_mult[i])
            for j in range(self.num_res_blocks):
                if j < num_res_blocks - 1:
                    layers.append(
                        ResnetBlockBigGANpp(
                            act=self.act, in_ch=in_ch, out_ch=out_ch, temb_dim=4 * nf
                        )
                    )
                else:
                    out_ch = int(nf * ch_mult[i + 1]) if i + 1 < len(ch_mult) else in_ch
                    layers.append(
                        ResnetBlockBigGANpp(
                            act=self.act,
                            in_ch=in_ch,
                            out_ch=out_ch,
                            temb_dim=4 * nf,
                            down=True,
                            factor=factor,
                        )
                    )
        self.input_branch = nn.ModuleList(layers)
        self.final_pooling_layer = nn.AvgPool2d(kernel_size=output_kernel)

        ### Latent encoder ###
        self._image_latent_size = self._latent_pixels * self._latent_pixels * self._latent_channels
        layers = []
        layers.append(nn.Linear(self._image_latent_size, hidden_size))
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.latent_branch = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, t, x, *args, **kwargs):
        ############ Time branch ############
        temb = self.time_branch[0](t)  # Gaussian Fourier Projection
        if self.conditioned:
            # Combine time embedding with conditionals if any
            temb = merge_conditional_time_branch(self, temb, *args)
        temb = self.time_branch[1](temb)
        temb = self.time_branch[2](self.act(temb))  # pre activation convention

        ############ Input branch ############
        if self.conditioned:
            # Combine input tensor with input tensors if any
            x = merge_conditional_input_branch(self, x, *args)
        h = self.input_layer(x)
        for block in self.input_branch:
            h = block(h, temb)
        h = self.final_pooling_layer(h)

        ############ Latent encoder ############
        h = h.view(-1, self._image_latent_size)  # flatten
        for layer in self.latent_branch:
            h = self.act(layer(h))
        return self.output_layer(h)
