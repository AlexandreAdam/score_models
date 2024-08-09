import torch
import torch.nn as nn
import numpy as np


class AnnealedScoreModel(nn.Module):

    def __init__(
        self, sde, approx_model, target_model, beta_scheme="linear", epsilon=0.01, **kwargs
    ):
        super().__init__()
        self.sde = sde
        self.approx_model = approx_model
        self.target_model = target_model
        self.beta_scheme = beta_scheme
        self.epsilon = epsilon

    def beta(self, t):
        T = (t - self.sde.t_min) / (self.sde.t_max - self.sde.t_min)
        if self.beta_scheme == "linear":
            return T
        elif self.beta_scheme == "square":
            return T**2
        elif self.beta_scheme == "sqrt":
            return torch.sqrt(T)
        elif "linear:" in self.beta_scheme:
            return int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T
        elif "sqrt:" in self.beta_scheme:
            return torch.sqrt(int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T)
        elif "sin:" in self.beta_scheme:
            i = int(self.beta_scheme[self.beta_scheme.find(":") + 1 :])
            return torch.where(T > 1 / i, torch.ones_like(T), torch.sin(i * T * torch.pi / 2.0))
        else:
            raise NotImplementedError(f"Unknown beta_scheme {self.beta_scheme}")

    def forward(self, t, x, **kwargs):
        B, *D = x.shape
        # Compute the weighted score for each model
        beta = torch.clamp(self.beta(kwargs.get("t_a", t)[0]), 0.0, 1.0)
        score = torch.zeros_like(x)
        if beta.item() > self.epsilon:
            score += self.approx_model(t, x, **kwargs) * beta
        if beta.item() < (1 - self.epsilon):
            score += self.target_model(t, x, **kwargs) * (1.0 - beta)
        return score * self.sde.sigma(t).view(-1, *[1] * len(D))
