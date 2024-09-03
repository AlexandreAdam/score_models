from typing import Callable, Union, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch import vmap
import numpy as np

from ..sde import SDE


class ConvolvedLikelihood(nn.Module):
    """
    Convolved likelihood approximation for the likelihood component of a
    posterior score model.

    Applies the convolved likelihood approximation as described in Adam et al.
    2022 Appendix A. Essentially this assumes that the posterior convolution may
    be approximately factored into the convolution of the likelihood and the
    prior separately. For a linear and gaussian likelihood model, this is exact,
    coming out to:

    .. math::

        p_t(y|x_t) = N(y|Ax_t, \Sigma_y + \\sigma_t^2 AA^T)

    We implement this as an energy model, where the energy is the negative log
    likelihood in the observation space. Autodiff then propagates the score to
    the model space.

    Args:
        sde: The SDE that the score model is associated with.
        y: The observation.
        Sigma_y: The observation covariance matrix. If 1D this is assumed to be the diagonal of the covariance matrix.
        x_shape: The shape of the model space.
        A: The linear operator relating the model space to the observation space.
        f: The function relating the model space to the observation space.
        AAT: The covariance of the linear operator.
        diag: Whether to use the diagonal approximation.
    """

    @torch.no_grad()
    def __init__(
        self,
        sde: SDE,
        y: Tensor,
        Sigma_y: Tensor,
        x_shape: Tuple[int],
        A: Optional[Tensor] = None,
        f: Optional[Callable] = None,
        AAT: Optional[Tensor] = None,
        diag: bool = False,
    ):
        assert (A is not None) != (f is not None), "Either A or f must be provided (not both)"
        super().__init__()
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.x_shape = x_shape
        self.y_shape = y.shape
        if A is not None:
            self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        else:
            self.A = A
        self.f = f
        if AAT is None:
            if A is None:
                A = torch.func.jacrev(f)(
                    torch.zeros(x_shape, dtype=y.dtype, device=y.device)
                ).reshape(np.prod(self.y_shape), np.prod(x_shape))
                if diag:
                    self.AAT = torch.sum(A**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = A @ A.T
            else:
                if diag:
                    self.AAT = torch.sum(self.A**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT

        self.diag = diag
        self.hyperparameters = {"nn_is_energy": True}

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        self.forward = self.diag_forward if value else self.full_forward

    def diag_forward(self, t, xt, *args, **kwargs):
        if self.f is None:
            r = self.y - (self.A @ xt.reshape(-1)).reshape(*self.y_shape)
        else:
            r = self.y - self.f(xt)
        sigma = 1 / (self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * torch.sum(r**2 * sigma).unsqueeze(0)
        return ll.unsqueeze(0) * self.sde.sigma(t)

    def _full_forward(self, t, xt, sigma):
        if self.f is None:
            r = self.y.reshape(-1) - self.A @ xt.reshape(-1)
        else:
            r = self.y.reshape(-1) - self.f(xt).reshape(-1)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll * self.sde.sigma(t)

    def full_forward(self, t, xt, *args, **kwargs):
        sigma = torch.linalg.inv(
            self.Sigma_y * self.sde.mu(t[0]) ** 2 + self.sde.sigma(t[0]) ** 2 * self.AAT
        )
        return vmap(self._full_forward, in_dims=(0, 0, None))(t, xt, sigma)
