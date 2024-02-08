import jax.numpy as jnp
import equinox as eqx
import numpy as np
import jax


def uniform_init(key, shape, scale=1e-2):
    return jax.random.uniform(key, shape, minval=-scale, maxval=scale)


class SelfAttentionBlock(eqx.Module):
    to_qkv: eqx.nn.Conv
    to_out: eqx.nn.Conv

    def __init__(self, channels, init_scale=1e-2, dimensions=2):
        conv = {1: eqx.nn.Conv1D, 2: eqx.nn.Conv2D, 3: eqx.nn.Conv3D}[dimensions]
        self.to_qkv = conv(channels, 3 * channels, kernel_size=1)
        self.to_out = conv(channels, channels, kernel_size=1)

        # Manually adjust weights after creation
        bound_qkv = 1 / channels**0.5
        self.to_qkv.weight = eqx.filter_jit(lambda _: True)(
            jax.nn.initializers.uniform(-bound_qkv, bound_qkv)
        )(self.to_qkv.weight.shape)
        self.to_qkv.bias = jnp.zeros_like(self.to_qkv.bias)

        bound_out = init_scale / channels**0.5
        self.to_out.weight = eqx.filter_jit(lambda _: True)(
            jax.nn.initializers.uniform(-bound_out, bound_out)
        )(self.to_out.weight.shape)
        self.to_out.bias = jnp.zeros_like(self.to_out.bias)

    def __call__(self, x):
        B, C, *D = x.shape
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=1)
        q = q.reshape(B, np.prod(D), C)
        k = k.reshape(B, C, np.prod(D))
        v = v.reshape(B, np.prod(D), C)

        w = jax.lax.batch_matmul(q, k) * (C ** (-0.5))
        w = eqx.nn.softmax(w, axis=-1)
        attention = jax.lax.batch_matmul(w, v)
        attention = attention.reshape(B, *D, C).transpose(
            (0, -1) + tuple(range(1, len(D) + 1))
        )
        return self.to_out(attention) + x


class ScaledAttentionLayer(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    to_out: eqx.nn.Linear

    def __init__(self, dimensions):
        self.query = eqx.nn.Linear(dimensions, dimensions)
        self.key = eqx.nn.Linear(dimensions, dimensions)
        self.value = eqx.nn.Linear(dimensions, dimensions)
        self.to_out = eqx.nn.Linear(dimensions, dimensions)

        # Manually adjust weights after creation
        bound = 1 / dimensions**0.5
        for layer in [self.query, self.key, self.value, self.to_out]:
            layer.weight = eqx.filter_jit(lambda _: True)(
                jax.nn.initializers.uniform(-bound, bound)
            )(layer.weight.shape)
            layer.bias = jnp.zeros_like(layer.bias)

    def __call__(self, query, context):
        B, C_out, D = query.shape
        query = self.query(query.reshape(B * C_out, D)).reshape(B, C_out, D)
        value = self.value(context.reshape(B * C_out, D)).reshape(B, C_out, D)
        scores = jax.lax.batch_matmul(query, context.transpose((0, 2, 1))) / D**0.5
        scores = eqx.nn.softmax(scores, axis=-1)
        attention = jax.lax.batch_matmul(scores, value)
        h = self.to_out(attention)
        return h
