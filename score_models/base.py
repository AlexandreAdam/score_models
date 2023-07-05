import torch
from torch.nn import Module
from score_models.sde import VESDE, VPSDE
from typing import Union
from .utils import load_architecture
from abc import ABC, abstractmethod
from tqdm import tqdm


class ScoreModelBase(Module, ABC):
    def __init__(self, model: Union[str, Module]=None, checkpoint_directory=None, **hyperparameters):
        super().__init__()
        if model is None or isinstance(model, str):
            model, hyperparams = load_architecture(checkpoint_directory, model=model, hyperparameters=hyperparameters)
            if "sde" not in hyperparams.keys():
                if "sigma_min" in hyperparams.keys():
                    hyperparams["sde"] = "vesde"
                elif "beta_min" in hyperparams.keys():
                    hyperparams["sde"] = "vpsde"
            if hyperparams["sde"].lower() == "vesde":
                sde = VESDE(sigma_min=hyperparams["sigma_min"], sigma_max=hyperparams["sigma_max"])
            elif hyperparams["sde"].lower() == "vpsde":
                sde = VPSDE(beta_min=hyperparams["beta_min"], beta_max=hyperparams["beta_max"])
            else:
                raise ValueError("sde parameters missing from hyperparameters")
        self.model = model
        self.sde = sde

    def forward(self, t, x):
        return self.score(t, x)
   
    @abstractmethod
    def score(self, t, x):
        ...

    @torch.no_grad()
    def sample(self, size, N: int):
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
    
    def fit(dataset,):
        return loss
