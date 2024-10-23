from typing import Optional, Literal

import torch
import torch.nn as nn

from ..layers import GaussianFourierProjection, ScaledAttentionLayer
from .conditional_branch import (
    validate_conditional_arguments,
    conditional_branch,
    merge_conditional_time_branch,
    merge_conditional_input_branch,
)
from ..utils import get_activation

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int = 100,
        layers: int = 4,
        fourier_scale: int = 16,
        activation: int = "silu",
        attention: bool = False,
        nn_is_energy: bool = False,
        output_activation: str = None,
        conditions: Optional[Literal["discrete", "continuous", "vector", "tensor"]] = None,
        condition_channels: Optional[tuple[int]] = None,
        condition_embeddings: Optional[tuple[int]] = None,
        **kwargs,
    ):
        """
        Multi-Layer Perceptron (MLP) neural network.

        Parameters:
        - channels (Optional[int]): Number of input channels. Default is None.
        - width (int): Number of units in each hidden layer. Default is 100.
        - layers (int): Number of hidden layers. Default is 2.
        - fourier_scale (int): Scale factor for Fourier features. Default is 16.
        - activation (str): Activation function to use. Default is "swish".
        - attention (bool): Whether to use attention mechanism. Default is False.
        - nn_is_energy (bool): Whether the neural network represents energy. Default is False.
        - output_activation (str): Activation function for the output layer. Default is None.
        - conditions (Optional[Literal["discrete", "continuous", "vector", "tensor"]]): Type of conditions. Default is None.
        - condition_channels (Optional[tuple[int]]): Channels for conditioning. Default is None.
        - condition_embeddings (Optional[tuple[int]]): Embeddings for conditioning. Default is None.
        - **kwargs: Additional keyword arguments.

        """
        super().__init__()
        validate_conditional_arguments(conditions, condition_embeddings, condition_channels)
        self.conditioned = conditions is not None
        self.condition_type = conditions
        self.condition_embeddings = condition_embeddings
        self.condition_channels = condition_channels
        self.hyperparameters = {
            "channels": channels,
            "width": width,
            "layers": layers,
            "fourier_scale": fourier_scale,
            "activation": activation,
            "attention": attention,
            "nn_is_energy": nn_is_energy,
            "output_activation": output_activation,
            "conditions": conditions,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings,
        }
        self.layers = layers
        self.attention = attention
        self.nn_is_energy = nn_is_energy
        if layers % 2 == 1:
            layers += 1

        ########### Conditional branch ###########
        if self.conditioned:
            total_time_channels, total_input_channels = conditional_branch(
                self,
                time_branch_channels=width,
                input_branch_channels=channels,
                condition_embeddings=condition_embeddings,
                condition_channels=condition_channels,
                fourier_scale=fourier_scale,
            )
        else:
            total_time_channels = width
            total_input_channels = channels

        ########### Time branch ###########
        time_branch = [
            GaussianFourierProjection(width, scale=fourier_scale),
            nn.Linear(total_time_channels, width),
            nn.Linear(width, width),
        ]
        self.time_branch = nn.ModuleList(time_branch)

        ########### Input branch ###########
        modules = []
        self.input_layer = nn.Linear(total_input_channels, width)
        for _ in range(layers):
            modules.append(nn.Linear(2*width, width))
        self.linear_layers = nn.ModuleList(modules)
        if self.attention:
            self.attention_layer = ScaledAttentionLayer(width)
        if nn_is_energy:
            self.output_layer = nn.Linear(2*width, 1)
            self.output_act = get_activation(output_activation)
        else:
            self.output_layer = nn.Linear(2*width, channels)
        self.act = get_activation(activation)

    def forward(self, t, x, *args, **kwargs):
        B, D = x.shape

        ########### Time branch ###########
        temb = self.time_branch[0](t)
        if self.conditioned:
            temb = merge_conditional_time_branch(self, temb, *args)
        for layer in self.time_branch[1:]:
            temb = self.act(layer(temb))

        ########### Input branch ###########
        if self.conditioned:
            x = merge_conditional_input_branch(self, x, *args)
        x = self.input_layer(x)
        for layer in self.linear_layers[:self.layers//2]:
            x = torch.cat([x, temb], dim=1)
            x = self.act(layer(x))
        if self.attention:
            context = temb.view(B, 1, -1)
            query = x.view(B, 1, -1)
            x = self.attention_layer(query, context).view(B, -1)
        for layer in self.linear_layers[self.layers//2:]:
            x = torch.cat([x, temb], dim=1)
            x = self.act(layer(x))
        x = torch.cat([x, temb], dim=1)
        out = self.output_layer(x)
        if self.nn_is_energy:
            out = self.output_act(out)
        return out
