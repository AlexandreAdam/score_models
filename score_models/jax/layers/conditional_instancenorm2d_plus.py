from jaxtyping import PRNGKeyArray
from typing import Optional, Union
import equinox as eqx
import jax.numpy as jnp


class ConditionalInstanceNorm2dPlus(eqx.Module):
    num_features: int
    bias: bool
    instance_norm: eqx.nn.LayerNorm
    embed: Union[eqx.nn.Embedding, eqx.nn.Linear]
    num_classes: Optional[int]

    def __init__(self, num_features, num_classes=None, bias=True, *, key: PRNGKeyArray):
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = eqx.nn.LayerNorm(shape=(num_features, 1, 1))

        if num_classes is None:
            embed_output_dim = num_features * 3 if bias else num_features * 2
            self.embed = eqx.nn.Linear(1, embed_output_dim, use_bias=bias, key=key)
            # if bias:
                # self.embed.bias = zeros(key=jax.random.PRNGKey(0), shape=self.embed.bias.shape)
        else:
            self.num_classes = num_classes
            embed_output_dim = num_features * 3 if bias else num_features * 2
            self.embed = eqx.nn.Embedding(num_classes, embed_output_dim)

        # Initialize weights
        # self.embed.weight = normal(stddev=0.02)(shape=self.embed.weight.shape, key=key)

    def __call__(self, x, condition):
        condition = condition[:, None] if self.num_classes is None else condition
        means = jnp.mean(x, axis=(2, 3), keepdims=True)
        m = jnp.mean(means, axis=1, keepdims=True)
        v = jnp.var(means, axis=1, keepdims=True)
        means_norm = (means - m) / (jnp.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        embed_out = self.embed(condition)
        gamma, alpha = (
            embed_out[:, : self.num_features],
            embed_out[:, self.num_features : self.num_features * 2],
        )
        h = h + means_norm * alpha[..., None, None]
        out = gamma.reshape(-1, self.num_features, 1, 1) * h
        if self.bias:
            beta = embed_out[:, self.num_features * 2 :]
            out = out + beta.reshape(-1, self.num_features, 1, 1)
        return out
