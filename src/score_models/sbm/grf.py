import torch
from torch import Tensor

from .energy_model import EnergyModel
from ..sde import SDE
from ..architectures import NullNet


class GRFEnergyModel(EnergyModel):
    """
    Gaussian random field score model.

    Computes the energy for a gaussian random field based on a provided power spectrum.

    Args:
        sde: The SDE that the score model is associated with.
        power_spectrum: The power spectrum of the Gaussian random field.
    """

    def __init__(self, sde: SDE, power_spectrum: Tensor, **kwargs):
        super().__init__(net=NullNet(isenergy=True), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        # Store the power spectrum
        self.power_spectrum = power_spectrum
        self.dims = power_spectrum.dim()
        if self.dims == 1:
            self.fft = torch.fft.fft
        elif self.dims == 2:
            self.fft = torch.fft.fft2
        else:
            raise ValueError("Only 1D and 2D power spectra are supported")

    def energy(self, t: Tensor, x: Tensor, *args, **kwargs):
        """GRF energy"""
        sigma_t = self.sde.sigma(t)
        mu_t = self.sde.mu(t)

        # Fourier Transform of the image
        fftkwargs = {"norm": "ortho"}
        if self.dims == 2:
            fftkwargs["s"] = x.shape[-2:]
        elif self.dims == 1:
            fftkwargs["n"] = x.shape[-1]
        image_ft = self.fft(x, **fftkwargs)

        # Compute squared magnitude of the Fourier coefficients
        magnitude_squared = torch.abs(image_ft) ** 2

        # Calculate negative log likelihood
        nll = 0.5 * torch.sum(
            (magnitude_squared / (mu_t**2 * self.power_spectrum + sigma_t**2)).real,
            dim=tuple(range(-self.dims, 0)),
        )
        return nll

    def unnormalized_energy(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Unnormalized energy is not defined for GRF models.")

    def reparametrized_score(self, t, x, *args, **kwargs):
        raise RuntimeError("Reparametrized score is not defined for GRF models.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("GRF models cannot yet be saved.")
    
    def load(self, *args, **kwargs):
        raise NotImplementedError("GRF models cannot yet be loaded.")
