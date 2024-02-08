import equinox as eqx
import jax


class ConditionalInstanceNorm2d(eqx.Module):
    instance_norm: eqx.nn.InstanceNorm
    embed: eqx.Module
    num_features: int
    bias: bool
    num_classes: int = None

    def __init__(self, num_features, num_classes=None, bias=True):
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = eqx.nn.InstanceNorm(
            num_features, use_running_average=False
        )
        if num_classes is None:
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Linear(1, embed_output_dim, use_bias=False)
        else:
            self.num_classes = num_classes
            embed_output_dim = num_features * 2 if bias else num_features
            self.embed = eqx.nn.Embedding(num_classes, embed_output_dim)

        # Initialize weights
        self.embed.weight = jax.nn.initializers.normal(stddev=0.02)(
            self.embed.weight.shape
        )
        if bias:
            self.embed.weight = self.embed.weight.at[:, num_features:].set(
                0
            )  # Set bias part to 0

    def __call__(self, x, condition):
        condition = condition[:, None] if self.num_classes is None else condition
        h = self.instance_norm(x)

        embed_out = self.embed(condition)
        if self.bias:
            gamma, beta = (
                embed_out[:, : self.num_features],
                embed_out[:, self.num_features :],
            )
            out = gamma.reshape(-1, self.num_features, 1, 1) * h + beta.reshape(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = embed_out
            out = gamma.reshape(-1, self.num_features, 1, 1) * h
        return out
