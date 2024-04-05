import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.nn.initializers import uniform, zeros
from jaxtyping import PRNGKeyArray
from jax import vmap
import jax


__all__ = ["SelfAttentionBlock", "ScaledAttentionLayer"]


# Not used in this implementation
def uniform_init(key, shape, scale=1e-2):
    return jax.random.uniform(key, shape, minval=-scale, maxval=scale)


class SelfAttentionBlock(eqx.Module):
    to_qkv: eqx.nn.Conv
    to_out: eqx.nn.Conv

    def __init__(self, channels, init_scale=1e-2, dimensions=2, *, key: PRNGKeyArray): 
        key_qkv, key_out = jax.random.split(key)

        conv = {1: eqx.nn.Conv1d, 2: eqx.nn.Conv2d, 3: eqx.nn.Conv3d}[dimensions] 
        self.to_qkv = conv(channels, 3 * channels, kernel_size=1, key=key_qkv)
        self.to_out = conv(channels, channels, kernel_size=1, key=key_out)

        # Init
        # bound_qkv = 1 / channels**0.5
        # self.to_qkv.weight = uniform(-bound_qkv, bound_qkv)(shape=self.to_qkv.weight.shape, key=key_qkv)
        # self.to_qkv.bias = zeros(shape=self.to_qkv.bias.shape, key=jax.random.PRNGKey(0))

        # bound_out = init_scale / channels**0.5
        # self.to_out.weight = uniform(-bound_out, bound_out)(shape=self.to_out.weight.shape, key=key_out)
        # self.to_out.bias = zeros(shape=self.to_out.bias.shape, key=jax.random.PRNGKey(0))

    def __call__(self, x):
        C, *D = x.shape
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=0)
        
        q = jnp.transpose(q.reshape(C, -1))
        k = k.reshape(C, -1)
        v = jnp.transpose(v.reshape(C, -1))

        w = jnp.matmul(q, k) * (C ** (-0.5))
        w = jax.nn.softmax(w, axis=-1)
        attention = jnp.matmul(w, v)
        attention = jnp.transpose(attention).reshape(C, *D)
        return self.to_out(attention) + x


class ScaledAttentionLayer(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    to_out: eqx.nn.Linear

    def __init__(self, dimensions, *, key: PRNGKeyArray):
        key_query, key_key, key_value, key_out = jax.random.split(key, 4)
        self.query = eqx.nn.Linear(dimensions, dimensions, key=key_query)
        self.key = eqx.nn.Linear(dimensions, dimensions, key=key_key)
        self.value = eqx.nn.Linear(dimensions, dimensions, key=key_value)
        self.to_out = eqx.nn.Linear(dimensions, dimensions, key=key_out)

        # Init
        # bound = 1 / dimensions**0.5
        # for layer, key in zip([self.query, self.key, self.value, self.to_out], [key_query, key_key, key_value, key_out]):
            # layer.weight = uniform(-bound, bound)(shape=layer.weight.shape, key=key)
            # layer.bias = zeros(shape=layer.bias.shape, key=jax.random.PRNGKey(0))

    def __call__(self, query, context):
        C_out, D = query.shape
        C_in, D = context.shape
        
        query = vmap(self.query)(query)
        value = vmap(self.value)(context)
        scores = jnp.matmul(query, context.transpose()) / D**0.5 # scaled attention matrix, QK^T / sqrt(d)
        scores = jax.nn.softmax(scores, axis=1)
        attention = jnp.matmul(scores, value)
        h = vmap(self.to_out)(attention)
        return h

