from jaxtyping import PRNGKeyArray
from typing import Callable, Optional
from ..layers import GaussianFourierProjection, ScaledAttentionLayer
from ..utils import get_activation
import jax
import jax.numpy as jnp
import equinox as eqx


__all__ = ["MLP"]


class MLP(eqx.Module):
    activation: Callable
    gaussian_fourier_projection: GaussianFourierProjection
    time_branch: list
    main_branch: list
    bottleneck_in: Optional[eqx.nn.Linear]
    bottleneck_out: Optional[eqx.nn.Linear]
    temb_to_bottleneck: Optional[eqx.nn.Linear]
    attention_layer: Optional[ScaledAttentionLayer]
    output_layer: eqx.nn.Linear
    output_activation: Callable

    def __init__(
        self,
        dimensions,
        units=100,
        layers=2,
        time_embedding_dimensions=32,
        embedding_scale=30,
        activation="swish",
        time_branch_layers=1,
        bottleneck=None,
        attention=False,
        nn_is_energy=False,
        output_activation=None,
        conditioning=["none"],
        conditioning_channels=None,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        key_proj, key_time_branch, key_main_branch, key_bottleneck, key_output = jax.random.split(key, 5)
        
        self.activation = get_activation(activation)
        self.gaussian_fourier_projection = GaussianFourierProjection(time_embedding_dimensions, scale=embedding_scale, key=key_proj)

        # Time branch layers
        key_time_layers = jax.random.split(key_time_branch, time_branch_layers)
        self.time_branch = [eqx.nn.Linear(time_embedding_dimensions, time_embedding_dimensions, key=key_time_layers[i])
            for i in range(time_branch_layers)
        ]

        # Main branch layers
        key_main_layers = jax.random.split(key_main_branch, layers)
        self.main_branch = [eqx.nn.Linear(dimensions + time_embedding_dimensions, units, key=key_main_layers[0])]
        self.main_branch += [
            eqx.nn.Linear(units, units if not bottleneck or i < layers - 1 else bottleneck, key=key_main_layers[i + 1])
            for i in range(layers - 1)
        ]

        if bottleneck and attention:
            key_in, key_out, key_temb_to_bottleneck, key_attention = jax.random.split(key_bottleneck, 4)
            self.bottleneck_in = eqx.nn.Linear(units, bottleneck, key=key_in)
            self.bottleneck_out = eqx.nn.Linear(bottleneck, units, key=key_out)
            self.temb_to_bottleneck = eqx.nn.Linear(time_embedding_dimensions, bottleneck, key=key_temb_to_bottleneck)
            self.attention_layer = ScaledAttentionLayer(bottleneck, key=key_attention)
        else:
            self.bottleneck_in = None
            self.bottleneck_out = None
            self.temb_to_bottleneck = None
            self.attention_layer = None

        # Output layer
        output_units = 1 if nn_is_energy else dimensions
        self.output_layer = eqx.nn.Linear(units, output_units, key=key_output)
        self.output_activation = get_activation(output_activation) if nn_is_energy else None

    def __call__(self, t, x):
        temb = self.gaussian_fourier_projection(t)
        for layer in self.time_branch:
            temb = self.activation(layer(temb))

        x = jnp.concatenate([x, temb], axis=1)
        for layer in self.main_branch:
            x = self.activation(layer(x))

        if self.bottleneck_in is not None:
            x = self.activation(self.bottleneck_in(x))

        if self.attention_layer is not None:
            temb = self.temb_to_bottleneck(temb)
            context = jnp.stack([x, temb], axis=1)
            x = self.attention_layer(
                x.reshape(-1, 1, self.bottleneck_in.out_features), context
            ).reshape(-1, self.bottleneck_out.out_features)

        if self.bottleneck_out is not None:
            x = self.activation(self.bottleneck_out(x))

        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
