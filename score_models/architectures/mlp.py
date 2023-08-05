import torch
import torch.nn as nn
from score_models.layers import GaussianFourierProjection, ScaledAttentionLayer
from score_models.utils import get_activation


class MLP(nn.Module):
    def __init__(
            self, 
            dimensions:int, 
            units:int=100, 
            layers:int=2, 
            time_embedding_dimensions:int =32, 
            embedding_scale:int=30,
            activation:int="swish",
            time_branch_layers:int=1,
            bottleneck:int=None,
            attention:bool=False,
            nn_is_energy:bool=False,
            output_activation:str=None,
            conditioning:list[str,...]=["none"],
            conditioning_channels:list[int,...]=None,
            **kwargs
            ):
        super().__init__()
        self.conditioned = False
        for c in conditioning:
            if c.lower() not in ["none", "input"]:
                raise ValueError(f"Conditioning must be in ['None', 'Input'], received {c}")
            if c.lower() != "none":
                self.conditioned = True
                if conditioning_channels is not None:
                    raise ValueError("conditioning_channels must be provided when the network is conditioned")
            elif c.lower() == "none" and self.conditioned:
                raise ValueError(f"Cannot have a mix of 'None' and other type of conditioning, received the list {conditioning}")
        self.hyperparameters = {
                "dimensions": dimensions,
                "units": units,
                "layers": layers,
                "time_embedding_dimensions": time_embedding_dimensions,
                "embedding_scale": embedding_scale,
                "activation": activation,
                "time_branch_layers": time_branch_layers,
                "nn_is_energy": nn_is_energy,
                "conditioning": conditioning
                }
        if nn_is_energy:
            self.hyperparameters.update({"output_activation": output_activation})
        self.time_branch_layers = time_branch_layers
        self.layers = layers
        self.nn_is_energy = nn_is_energy
        t_dim = time_embedding_dimensions
        if layers % 2 == 1:
            layers += 1
        # time embedding branch
        modules = [GaussianFourierProjection(t_dim, scale=embedding_scale)]
        for _ in range(time_branch_layers):
            modules.append(nn.Linear(t_dim, t_dim))
        # main branch
        modules.append(nn.Linear(dimensions+t_dim, units))
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
            self.output_layer = nn.Linear(units, dimensions)
        self.act = get_activation(activation)
        self.all_modules = nn.ModuleList(modules)
    
    def forward(self, t, x):
        B, D = x.shape
        modules = self.all_modules
        temb = modules[0](t)
        i = 1
        for _ in range(self.time_branch_layers):
            temb = self.act(modules[i](temb))
            i += 1
        x = torch.cat([x, temb], dim=1)
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

