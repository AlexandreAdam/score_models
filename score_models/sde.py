""" DE classes, Reverse SDE, and VE SDEs.
Copied from Yang Song's repo, with slight modifications

"""
import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super().__init__()

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.
        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t, N=1000):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.
        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
          f, G
        """
        dt = 1 / N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                super().__init__()
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, N=1000):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, N)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50.):
        """Construct a Variance Exploding SDE.
        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t, N=1000):
        """SMLD(NCSN) discretization."""
        sigma = self.sigma(t)
        adjacent_sigma = self.sigma(t - 1/N)
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
