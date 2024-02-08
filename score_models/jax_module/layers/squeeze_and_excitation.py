import jax.numpy as jnp
import jax
import equinox as eqx
"""
Based on the paper
    "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
Taken from original implementation
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/12dcf945a6359301d63d1e0da3708cd0f0590b19/spectral_normalization.py
"""

def l2normalize(v, eps=1e-12):
    return v / (jnp.linalg.norm(v) + eps)

class SpectralNorm(eqx.Module):
    module: eqx.Module
    name: str = 'weight'
    power_iterations: int = 1
    u: jnp.ndarray = eqx.static_field()
    v: jnp.ndarray = eqx.static_field()
    w_bar: jnp.ndarray = eqx.static_field()

    def __init__(self, module, name='weight', power_iterations=1, *, key):
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params(key)

    def _make_params(self, key):
        w = getattr(self.module, self.name)
        height = w.shape[0]
        width = w.reshape(height, -1).shape[1]

        u_key, v_key = jax.random.split(key)
        u = jax.random.normal(u_key, (height,))
        v = jax.random.normal(v_key, (width,))
        u = l2normalize(u)
        v = l2normalize(v)
        w_bar = w

        self.u = u
        self.v = v
        self.w_bar = w_bar

    def update_u_v(self):
        u = self.u
        v = self.v
        w = self.w_bar

        height = w.shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(jnp.dot(w.reshape(height, -1).T, u))
            u = l2normalize(jnp.dot(w.reshape(height, -1), v))

        sigma = jnp.dot(u, jnp.dot(w.reshape(height, -1), v))
        setattr(self.module, self.name, w / sigma)

    def __call__(self, *args, **kwargs):
        self.update_u_v()
        return self.module(*args, **kwargs)


