from jaxtyping import PRNGKeyArray
from jax.nn.initializers import zeros, normal
from typing import Optional, Union
import equinox as eqx
import jax


class ConditionalBatchNorm2d(eqx.Module):
    num_features: int
    bias: bool
    num_classes: Optional[int]
    bn: eqx.nn.BatchNorm
    embed: Union[eqx.nn.Linear, eqx.nn.Embedding]

    def __init__(self, num_features, num_classes=None, bias=True, *, key: PRNGKeyArray):
        self.num_features = num_features
        self.bias = bias
        self.bn = eqx.nn.BatchNorm(input_size=num_features, axis_name='batch')
        if num_classes is None:
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Linear(1, embed_output_dim, use_bias=bias, key=key)
            if bias:
                self.embed.bias = zeros(key=jax.random.PRNGKey(0), shape=self.embed.bias.shape)
            
        else:
            self.num_classes = num_classes
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Embedding(num_classes, embed_output_dim)

        # Initialize weights
        self.embed.weight = normal(stddev=0.02)(shape=self.embed.weight.shape, key=key)

    def __call__(self, x, condition):
        condition = condition[:, None] if self.num_classes is None else condition
        out = self.bn(x)
        embed_out = self.embed(condition)
        if self.bias:
            gamma, beta = (
                embed_out[:, : self.num_features],
                embed_out[:, self.num_features :],
            )
            out = gamma.reshape(-1, self.num_features, 1, 1) * out + beta.reshape(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = embed_out
            out = gamma.reshape(-1, self.num_features, 1, 1) * out
        return out
