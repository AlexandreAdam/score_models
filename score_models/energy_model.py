import torch
from torch.nn import Module
from score_models.sde import VESDE, VPSDE
from typing import Union
from functorch import grad, vmap
from .utils import load_architecture
from tqdm import tqdm

class EnergyModel(Module):
    def __init__(self, model: Union[str, Module]=None, checkpoint_directory=None):
        if model is None or isinstance(model, str):
            if checkpoint_directory is not None:
                model, hyperparams = load_architecture(checkpoints_directory, model=model)
            else:
                raise ValueError("checkpoint directory must be specified")
            if "sde" not in hyperparams.keys():
                if "sigma_min" in hyperparams.keys():
                    hyperparams["sde"] = "vesde"
                elif "beta_min" in hyperparams.keys():
                    hyperparams["sde"] = "vpsde"
            elif hyperparams["sde"].lower() == "vesde":
                sde = VESDE(sigma_min=hyperparams["sigma_min"], sigma_max=hyperparams["sigma_max"])
            elif hyperparams["sde"].lower() == "vpsde":
                sde = VPSDE(beta_min=hyperparams["beta_min"], beta_max=hyperparams["beta_max"])
            else:
                raise ValueError("sde parameters missing from hyperparameters")
        self.model = model
        self.sde = sde

    def energy(self, t, x):
        _, *D = x.shape
        return 0.5 * torch.sum((x - self.model(t, x))**2, dim=list(range(1, 1+len(D))))
    
    def score(self, t, x):
        return vmap(grad(self.energy, argnums=1))(t, x)

    @torch.no_grad()
    def sample(self, size, N: int = 1000):
        assert size[1] == self.model.channels
        assert len(size) == self.model.dimensions + 2
        assert N > 0
        # A simple Euler-Maruyama integration of VESDE
        x = self.sde.prior(size).to(self.device)
        dt = -1.0 / N
        t = torch.ones(size[0]).to(self.device)
        for _ in tqdm(range(N)):
            t += dt
            drift = self.sde.drift(t, x)
            diffusion = self.sde.diffusion(t, x)
            score = self.score(t, x)
            drift = drift - diffusion**2 * score
            z = torch.randn_like(x)
            x_mean = x + drift * dt
            x = x_mean + diffusion * (-dt)**(1/2) * z
        return x_mean

