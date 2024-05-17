import functools as ft
# import warnings
# from collections.abc import Sequence
from typing import Optional, overload, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from equinox._custom_types import sentinel
from equinox._misc import left_broadcast_to
from equinox import field, Module
from equinox.nn import State

__all__ = ["GroupNorm"]


class GroupNorm(Module):
    r"""
    Splits the first dimension ("channels") into groups of fixed size. Computes a mean
    and standard deviation over the contents of each group, and uses these to normalise
    the group. Optionally applies a channel-wise affine transformation afterwards.

    Given an input array $x$ of shape `(channels, ...)`, this layer splits this up into
    `groups`-many arrays $x_i$ each of shape `(channels/groups, ...)`, and for each one
    computes

    $$\frac{x_i - \mathbb{E}[x_i]}{\sqrt{\text{Var}[x_i] + \varepsilon}} * \gamma_i + \beta_i$$

    where $\gamma_i$, $\beta_i$ have shape `(channels/groups,)` if
    `channelwise_affine=True`, and $\gamma = 1$, $\beta = 0$ if
    `channelwise_affine=False`.

    ??? cite
        [Group Normalization](https://arxiv.org/abs/1803.08494)

        ```bibtex
        @article{wu2018group,
            author={Yuxin Wu and Kaiming He},
            title={Group Normalization},
            year={2018},
            journal={arXiv:1803.08494},
        }
        ```
    """  # noqa: E501

    groups: int = field(static=True)
    channels: Optional[int] = field(static=True)
    eps: float = field(static=True)
    channelwise_affine: bool = field(static=True)
    weight: Optional[Array]
    bias: Optional[Array]

    def __init__(
        self,
        groups: int,
        channels: Optional[int] = None,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
    ):
        """**Arguments:**

        - `groups`: The number of groups to split the input into.
        - `channels`: The number of input channels. May be left unspecified (e.g. just
            `None`) if `channelwise_affine=False`.
        - `eps`: Value added to denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable affine parameters.
        """
        if (channels is not None) and (channels % groups != 0):
            raise ValueError("The number of groups must divide the number of channels.")
        if (channels is None) and channelwise_affine:
            raise ValueError(
                "The number of channels should be specified if "
                "`channelwise_affine=True`"
            )
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.weight = jnp.ones(channels) if channelwise_affine else None
        self.bias = jnp.zeros(channels) if channelwise_affine else None

    @overload
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, State]:
        ...

    def __call__(
        self, x: Array, state: State = sentinel, *, key: Optional[PRNGKeyArray] = None
    ) -> Union[Array, tuple[Array, State]]:
        """**Arguments:**

        - `x`: A JAX array of shape `(channels, ...)`.
        - `state`: Ignored; provided for interchangability with the
            [`equinox.nn.BatchNorm`][] API.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The output is a JAX array of shape `(channels, ...)`.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned. The state
        is passed through unchanged. If `state` is not passed, then just the output is
        returned.
        """
        channels = x.shape[0]
        
        y = x.reshape(self.groups, channels // self.groups, *x.shape[1:])
        mean = jax.vmap(ft.partial(jnp.mean, keepdims=True))(y)
        variance = jax.vmap(ft.partial(jnp.var, keepdims=True, ddof=0))(y)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        
        out = (y - mean) * inv
        out = out.reshape(x.shape)
        if self.channelwise_affine:
            weight = left_broadcast_to(self.weight, out.shape)  # pyright: ignore
            bias = left_broadcast_to(self.bias, out.shape)  # pyright: ignore
            out = weight * out + bias
        if state is sentinel:
            return out
        else:
            return out, state

