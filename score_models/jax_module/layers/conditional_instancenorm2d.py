import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
from jax.nn.initializers import zeros, normal


class ConditionalInstanceNorm2d(eqx.Module):
    def __init__(self, num_features, num_classes=None, bias=True, *, key: PRNGKeyArray):
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = eqx.nn.LayerNorm(shape=(num_features, 1, 1))
        if num_classes is None:
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Linear(1, embed_output_dim, use_bias=bias, key=key)
            if bias:
                self.embed.bias = zeros(shape=self.embed.bias.shape, key=jax.random.PRNGKey(0)
        else:
            self.num_classes = num_classes
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Embedding(num_classes, embed_output_dim)
        self.embed.weight = normal(stddev=0.02)(self.embed.weight.shape, key=key)

    def __call__(self, x, condition):
        condition = condition[:, None] if self.num_classes is None else condition
        h = self.instance_norm(x)

        embed_out = self.embed(condition)
        if self.bias:
            gamma, beta = (embed_out[:, : self.num_features], embed_out[:, self.num_features :],)
            out = gamma.reshape(-1, self.num_features, 1, 1) * h + beta.reshape(-1, self.num_features, 1, 1)
        else:
            gamma = embed_out
            out = gamma.reshape(-1, self.num_features, 1, 1) * h
        return out
