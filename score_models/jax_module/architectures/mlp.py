import jax.numpy as jnp
import equinox as eqx
from ..layers import GaussianFourierProjection, ScaledAttentionLayer
from ..utils import get_activation


class MLP(eqx.Module):
    gaussian_fourier_projection: eqx.Module
    time_branch: list
    main_branch: list
    bottleneck_in: eqx.nn.Linear = None
    bottleneck_out: eqx.nn.Linear = None
    temb_to_bottleneck: eqx.nn.Linear = None
    attention_layer: ScaledAttentionLayer = None
    output_layer: eqx.nn.Linear
    act: Callable

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
    ):
        super().__init__()
        self.act = get_activation(activation)
        self.gaussian_fourier_projection = GaussianFourierProjection(
            time_embedding_dimensions, scale=embedding_scale
        )

        # Time branch layers
        self.time_branch = [
            eqx.nn.Linear(time_embedding_dimensions, time_embedding_dimensions)
            for _ in range(time_branch_layers)
        ]

        # Main branch layers
        self.main_branch = [
            eqx.nn.Linear(dimensions + time_embedding_dimensions, units)
        ]
        self.main_branch += [
            eqx.nn.Linear(
                units, units if not bottleneck or i < layers - 1 else bottleneck
            )
            for i in range(layers - 1)
        ]

        if bottleneck and attention:
            self.bottleneck_in = eqx.nn.Linear(units, bottleneck)
            self.bottleneck_out = eqx.nn.Linear(bottleneck, units)
            self.temb_to_bottleneck = eqx.nn.Linear(
                time_embedding_dimensions, bottleneck
            )
            self.attention_layer = ScaledAttentionLayer(bottleneck)

        # Output layer
        output_units = 1 if nn_is_energy else dimensions
        self.output_layer = eqx.nn.Linear(units, output_units)
        self.output_activation = (
            get_activation(output_activation) if nn_is_energy else None
        )

    def __call__(self, t, x):
        temb = self.gaussian_fourier_projection(t)
        for layer in self.time_branch:
            temb = self.act(layer(temb))

        x = jnp.concatenate([x, temb], axis=1)
        for layer in self.main_branch:
            x = self.act(layer(x))

        if self.bottleneck_in is not None:
            x = self.act(self.bottleneck_in(x))

        if self.attention_layer is not None:
            temb = self.temb_to_bottleneck(temb)
            context = jnp.stack([x, temb], axis=1)
            x = self.attention_layer(
                x.reshape(-1, 1, self.bottleneck_in.out_features), context
            ).reshape(-1, self.bottleneck_out.out_features)

        if self.bottleneck_out is not None:
            x = self.act(self.bottleneck_out(x))

        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
