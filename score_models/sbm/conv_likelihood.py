from typing import Callable, Union, Tuple, Optional
import torch
from torch import Tensor
from torch import vmap
import numpy as np

from ..sde import SDE
from .energy_model import EnergyModel
from ..architectures import NullNet


class ConvolvedLikelihood(EnergyModel):
    """
    Convolved likelihood approximation for the likelihood component of a
    posterior score model.

    Applies the convolved likelihood approximation as described in Adam et al.
    2022 Appendix A. Essentially this assumes that the posterior convolution may
    be approximately factored into the convolution of the likelihood and the
    prior separately. For a linear and gaussian likelihood model, this is exact,
    coming out to:

    .. math::

        p_t(y|x_t) = N(y|Ax_t, \\Sigma_y + \\sigma_t^2 AA^T)

    We implement this as an energy model, where the energy is the negative log
    likelihood in the observation space. Autodiff then propagates the score to
    the model space.

    Args:
        sde: The SDE that the score model is associated with.
        y: The observation.
        Sigma_y: The observation covariance matrix. If 1D this is assumed to be the diagonal of the covariance matrix (and you should have diag=True).
        x_shape: The shape of the model space.
        A: The linear operator relating the model space to the observation space. May be a tensor or a function.
        AAT: The covariance of the linear operator. With A as a matrix then this is A @ A.T and should have the same shape as Sigma_y.
        diag: Whether to use the diagonal approximation.
    """

    @torch.no_grad()
    def __init__(
        self,
        sde: SDE,
        y: Tensor,
        Sigma_y: Tensor,
        x_shape: Tuple[int],
        A: Union[Tensor, Callable] = None,
        AAT: Optional[Tensor] = None,
        diag: bool = False,
        **kwargs,
    ):
        super().__init__(net=NullNet(isenergy=True), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.x_shape = x_shape
        self.y_shape = y.shape

        if isinstance(A, Tensor):
            self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        else:
            self.A = A

        if AAT is None:
            if callable(A):
                Amatrix = torch.func.jacrev(A)(
                    torch.zeros(x_shape, dtype=y.dtype, device=y.device)
                ).reshape(np.prod(self.y_shape), np.prod(x_shape))
                if diag:
                    self.AAT = torch.sum(Amatrix**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = Amatrix @ Amatrix.T
            else:
                if diag:
                    self.AAT = torch.sum(self.A**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT
        self.diag = diag

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        self.energy = self.diag_energy if value else self.full_energy

    def diag_energy(self, t, xt, *args, sigma, **kwargs):
        if callable(self.A):
            r = self.y - self.A(xt)
        else:
            r = self.y - (self.A @ xt.reshape(-1)).reshape(*self.y_shape)

        ll = 0.5 * torch.sum(r**2 * sigma)
        return ll

    def _full_forward(self, t, xt, sigma):
        if callable(self.A):
            r = self.y.reshape(-1) - self.A(xt).reshape(-1)
        else:
            r = self.y.reshape(-1) - self.A @ xt.reshape(-1)
        ll = 0.5 * (r @ sigma @ r.reshape(1, -1).T)
        return ll.squeeze(0)

    def full_energy(self, t, xt, *args, sigma, **kwargs):

        return vmap(self._full_forward, in_dims=(0, 0, None))(t, xt, sigma)

    def score(self, t, x, *args, **kwargs):
        # Compute sigma once per time step
        if self.diag:
            sigma = 1 / (self.Sigma_y + self.sde.sigma(t[0]) ** 2 * self.AAT)
        else:
            sigma = torch.linalg.inv(
                self.Sigma_y * self.sde.mu(t[0]) ** 2 + self.sde.sigma(t[0]) ** 2 * self.AAT
            )
        return super().score(t, x, *args, sigma=sigma, **kwargs)

    def unnormalized_energy(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Unnormalized energy should not be called for GRF models.")

    def reparametrized_score(self, t, x, *args, **kwargs):
        raise RuntimeError("Reparametrized score should not be called for GRF models.")
