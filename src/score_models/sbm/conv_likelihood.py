from typing import Callable, Union, Tuple, Optional
from torch import Tensor
from torch import vmap
import torch
import numpy as np

from .energy_model import EnergyModel
from ..sde import SDE
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
        Sigma_y: The observation covariance matrix. If ``Sigma_y.shape == y.shape`` this is assumed to be the diagonal of the covariance matrix.
        A: The linear operator relating the model space to the observation space. May be a tensor or a function.
        AAT: The covariance of the linear operator. With A as a matrix then this is A @ A.T and should have the same shape as Sigma_y.
        x_shape: The shape of the model space. This must be provided if A is a function and AAT is not provided.
    """

    @torch.no_grad()
    def __init__(
        self,
        sde: SDE,
        y: Tensor,
        Sigma_y: Tensor,
        A: Union[Tensor, Callable],
        AAT: Optional[Tensor] = None,
        x_shape: Optional[Tuple[int]] = None,
        **kwargs,
    ):
        super().__init__(net=NullNet(isenergy=True), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.y_shape = y.shape
        self.A = A
        self.x_shape = x_shape
        self.diag = self.y.shape == self.Sigma_y.shape

        if AAT is None:
            if callable(A):
                assert (
                    x_shape is not None
                ), "x_shape must be provided if A is a function and AAT is not provided."
                Amatrix = torch.func.jacrev(A)(
                    torch.zeros(x_shape, dtype=y.dtype, device=y.device)
                ).reshape(np.prod(self.y_shape), np.prod(x_shape))
                if self.diag:
                    self.AAT = torch.sum(Amatrix**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = Amatrix @ Amatrix.T
            else:
                if self.diag:
                    self.AAT = torch.sum(self.A**2, dim=1).reshape(*self.Sigma_y.shape)
                else:
                    self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT

        assert self.AAT.shape == self.Sigma_y.shape, "AAT must have the same shape as Sigma_y"

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
            r = self.y - (self.A @ xt.reshape(-1, 1)).reshape(*self.y_shape)
        nll = 0.5 * torch.sum(r**2 * sigma)
        return nll

    def _full_forward(self, t, xt, sigma):
        if callable(self.A):
            r = self.y.reshape(-1, 1) - self.A(xt).reshape(-1, 1)
        else:
            r = self.y.reshape(-1, 1) - self.A @ xt.reshape(-1, 1)
        nll = 0.5 * (r.T @ sigma @ r)
        return nll.squeeze()

    def full_energy(self, t, xt, *args, sigma, **kwargs):
        return vmap(self._full_forward, in_dims=(0, 0, None))(t, xt, sigma)

    def score(self, t, x, *args, **kwargs):
        # Compute sigma once per time step
        sigma = self.Sigma_y * self.sde.mu(t[0]) ** 2 + self.sde.sigma(t[0]) ** 2 * self.AAT
        sigma = 1 / sigma if self.diag else torch.linalg.inv(sigma)
        return super().score(t, x, *args, sigma=sigma, **kwargs)

    def unnormalized_energy(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Unnormalized energy is not defined for analytic models.")

    def reparametrized_score(self, t, x, *args, **kwargs):
        raise RuntimeError("Reparametrized score is not defined for analytic models.")
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("ConvolvedLikelihood models cannot yet be saved.")
    
    def load(self, *args, **kwargs):
        raise NotImplementedError("ConvolvedLikelihood models cannot yet be loaded.")
