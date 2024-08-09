import torch
import torch.nn as nn


class GRFScoreModel(nn.Module):
    """
    Gaussian random field score model.

    Computes the energy for a gaussian random field based on a provided power spectrum.

    Args:
        sde: The SDE that the score model is associated with.
        power_spectrum: The power spectrum of the Gaussian random field.
    """

    def __init__(self, sde, power_spectrum, dims=2):
        super().__init__()
        self.sde = sde
        # Store the power spectrum
        self.power_spectrum = power_spectrum
        self.dims = dims
        if dims == 1:
            self.fft = torch.fft.fft
        elif dims == 2:
            self.fft = torch.fft.fft2
        else:
            raise ValueError("Only 1D and 2D images are supported")
        self.hyperparameters = {"nn_is_energy": True}

    def forward(self, t, x, **kwargs):
        t_scale = self.sde.sigma(t)
        t_mu = self.sde.mu(t)

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
            (magnitude_squared / (t_mu**2 * self.power_spectrum + t_scale**2)).real,
            dim=tuple(range(-self.dims, 0)),
        )
        return nll.unsqueeze(1) * t_scale
