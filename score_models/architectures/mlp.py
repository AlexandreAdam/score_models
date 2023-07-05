import torch
import torch.nn as nn
from score_models.layers import GaussianFourierProjection, SelfAttentionBlock
from score_models.utils import get_activation


class MLP(nn.Module):
    def __init__(
            self, 
            input_dimensions, 
            units=100, 
            layers=2, 
            time_embedding_dimensions=32, 
            embedding_scale=30,
            activation="swish",
            time_branch_layers=1,
            bottleneck=None,
            attention=False,
            **kwargs
            ):
        super().__init__()
        self.hyperparameters = {
                "input_dimensions": input_dimensions,
                "units": units,
                "layers": layers,
                "time_embedding_dimensions": time_embedding_dimensions,
                "embedding_scale": embedding_scale,
                "activation": activation,
                "time_branch_layers": time_branch_layers
                }
        self.time_branch_layers = time_branch_layers
        self.layers = layers
        t_dim = time_embedding_dimensions
        if layers % 2 == 1:
            layers += 1
        # time embedding branch
        modules = [GaussianFourierProjection(time_embedding_dimensions, scale=embedding_scale)]
        for _ in range(time_branch_layers):
            modules.append(nn.Linear(t_dim, t_dim))
        # main branch
        modules.append(nn.Linear(input_dimensions+t_dim, units))
        if bottleneck is not None:
            assert isinstance(bottleneck, int)
            self.bottleneck = bottleneck
            self.bottleneck_in = nn.Linear(units, bottleneck)
            self.bottleneck_out = nn.Linear(bottleneck, units)
        else:
            self.bottleneck = 0
        self.attention = attention
        if self.attention:
            self.attention_layer = SelfAttentionBlock(bottleneck, dimensions=1)
        for _ in range(layers):
            modules.append(nn.Linear(units, units))
        self.output_layer = nn.Linear(units, input_dimensions)
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
            x = self.attention_layer(x.view(B, self.bottleneck, 1)).view(B, self.bottleneck)
        if self.bottleneck:
            x = self.act(self.bottleneck_out(x))
        for _ in range(self.layers//2):
            x = self.act(modules[i](x))
            i += 1
        return self.output_layer(x)

