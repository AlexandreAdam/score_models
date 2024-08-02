from typing import Optional, Literal

import torch
import torch.nn as nn

from ..layers import (
    GaussianFourierProjection, 
    ScaledAttentionLayer
    )
from .conditional_branch import (
    validate_conditional_arguments,
    conditional_branch,
    merge_conditional_time_branch,
    merge_conditional_input_branch
    )
from ..utils import get_activation

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
            self, 
            channels: Optional[int] = None,
            units: int = 100,
            layers: int = 2,
            time_branch_channels: int = 32,
            time_branch_layers: int = 1,
            fourier_scale: int = 16,
            activation: int = "swish",
            bottleneck: Optional[int] = None,
            attention: bool = False,
            nn_is_energy: bool = False,
            output_activation: str = None,
            conditions: Optional[Literal["discrete", "continuous", "vector", "tensor"]] = None,
            condition_channels: Optional[tuple[int]] = None,
            condition_embeddings: Optional[tuple[int]] = None,
            **kwargs
            ):
        """
        Multi-Layer Perceptron (MLP) neural network.

        Parameters:
        - channels (Optional[int]): Number of input channels. Default is None.
        - units (int): Number of units in each hidden layer. Default is 100.
        - layers (int): Number of hidden layers. Default is 2.
        - time_branch_channels (int): Number of channels in the time branch. Default is 32.
        - time_branch_layers (int): Number of layers in the time branch. Default is 1.
        - fourier_scale (int): Scale factor for Fourier features. Default is 16.
        - activation (str): Activation function to use. Default is "swish".
        - bottleneck (Optional[int]): Number of units in the bottleneck layer. Default is None.
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
        # Some backward compatibility
        if "embedding_scale" in kwargs:
            fourier_scale = kwargs["embedding_scale"]
        if "time_embedding_dimensions" in kwargs:
            time_branch_channels = kwargs["time_embedding_dimensions"]
        if channels is None:
            if "dimensions" in kwargs:
                channels = kwargs["dimensions"]
            else:
                raise ValueError("You must provide a 'channels' argument to initialize the MLP architecture.")
        self.hyperparameters = {
                "channels": channels,
                "units": units,
                "layers": layers,
                "time_branch_channels": time_branch_channels,
                "fourier_scale": fourier_scale,
                "activation": activation,
                "time_branch_layers": time_branch_layers,
                "botleneck": bottleneck,
                "attention": attention,
                "nn_is_energy": nn_is_energy,
                "output_activation": output_activation,
                "conditions": conditions,
                "condition_channels": condition_channels,
                "condition_embeddings": condition_embeddings,
                }
        self.time_branch_layers = time_branch_layers
        self.layers = layers
        self.nn_is_energy = nn_is_energy
        if layers % 2 == 1:
            print(f"Number of layers must be an even number for this architecture. Adding one more layer...")
            layers += 1

        ########### Conditional branch ###########
        if self.conditioned:
            total_time_channels, total_input_channels = conditional_branch(
                    self,
                    time_branch_channels=time_branch_channels,
                    input_branch_channels=channels,
                    condition_embeddings=condition_embeddings,
                    condition_channels=condition_channels,
                    fourier_scale=fourier_scale
                    ) # This method attach a Module list to self.conditional_branch
        else:
            total_time_channels = time_branch_channels
            total_input_channels = channels
        #########################################
            
        ########### Time branch ###########
        t_dim = time_branch_channels
        modules = [GaussianFourierProjection(t_dim, scale=fourier_scale), # Time embedding
                   nn.Linear(total_time_channels, t_dim) # Compress the signal from time index and the other conditionals if any
                   ]
        for _ in range(time_branch_layers - 1):
            modules.append(nn.Linear(t_dim, t_dim))
        ###################################
        
        ########### Input branch ###########
        modules.append(nn.Linear(total_input_channels + t_dim, units))
        if bottleneck is not None:
            assert isinstance(bottleneck, int)
            self.bottleneck = bottleneck
            self.bottleneck_in = nn.Linear(units, bottleneck)
            self.bottleneck_out = nn.Linear(bottleneck, units)
        else:
            self.bottleneck = 0
        self.attention = attention
        if self.attention:
            self.temb_to_bottleneck = nn.Linear(t_dim, bottleneck)
            self.attention_layer = ScaledAttentionLayer(bottleneck)
        for _ in range(layers):
            modules.append(nn.Linear(units, units))
        if nn_is_energy:
            self.output_layer = nn.Linear(units, 1)
            self.output_act = get_activation(output_activation)
        else:
            self.output_layer = nn.Linear(units, channels)
        self.act = get_activation(activation)
        self.all_modules = nn.ModuleList(modules)
        ###################################
    
    def forward(self, t, x, *args):
        B, D = x.shape
        modules = self.all_modules
        
        # Time branch
        temb = modules[0](t)
        if self.conditioned:
            temb = merge_conditional_time_branch(self, temb, *args)
        i = 1
        for _ in range(self.time_branch_layers):
            temb = self.act(modules[i](temb))
            i += 1

        # Input branch
        x = torch.cat([x, temb], dim=1)
        if self.conditioned:
            x = merge_conditional_input_branch(self, x, *args)
        x = modules[i](x)
        i += 1
        for _ in range(self.layers//2):
            x = self.act(modules[i](x))
            i += 1
        if self.bottleneck:
            x = self.act(self.bottleneck_in(x))
        if self.attention:
            temb = self.temb_to_bottleneck(temb)
            context = torch.stack([x, temb], dim=1)
            x = self.attention_layer(x.view(B, 1, -1), context).view(B, -1)
        if self.bottleneck:
            x = self.act(self.bottleneck_out(x))
        for _ in range(self.layers//2):
            x = self.act(modules[i](x))
            i += 1
        out = self.output_layer(x)
        if self.nn_is_energy:
            out = self.output_act(out)
        return out
