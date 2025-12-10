from typing import Optional

import torch
import copy
import numpy as np
from contextlib import contextmanager

__all__ = ["EMA"]

def ema_length_to_gamma(sigma_rel):
    """ 
    Algorithm 2 from Karras et al. 2024
    sigma_rel and ema_length are the same thing 
    """
    t = sigma_rel ** (-2)
    gamma = np.roots([1, 7, 16 - t, 12 - t]). real.max()
    return gamma


class EMA:
    """
    Exponential Moving Average (EMA) context manager of an online model produced by 
    gradient descent. The EMA model is updated with the decay parameter beta
        
        theta_t = (1 - beta) * theta_h + beta * theta_t
    
    I use theta_t for the Tortoise model (EMA) and theta_h for the Hare model (online).
    In a reformulation of the EMA (Karras et al. 2024), the decay parameter 
    is allowed to change over time, where

        beta(t) = (1 - dt/t) ** gamma
    
    When we choose to update the EMA after a certain number of steps, dt will take a value 
    other than 1, equal to the interval between updates.

    In addition, this class implement the soft reset mechanism (Noukhovitch et al. 2023, https://arxiv.org/abs/2312.07551), 
    which has been shown many time to improve training, especially in continuous learning scenarios.
    See for example the Hare and Tortoise paper (https://arxiv.org/abs/2406.02596)
    and the Switch EMA paper (https://arxiv.org/abs/2402.09240).
    
    For traditional EMA we employ a warmup strategy to avoid putting weight on the initial random network. 
    This idea is taken from lucidrains/ema_pytorch implementation. See also favel/torch_ema for a useful implementation.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            beta: Optional[float] = None,
            sigma_rel: Optional[float] = None,
            inverse_gamma: float = 1.0,         # Warmup schedule for classical EMA
            power: float = 2/3,                 # Warmup schedule for classical EMA
            start_update_after_step: int = 0,   # Delay update of traditional EMA
            update_every: int = 1,              # Update for the EMA
            ):
        self.step = 0

        # Schedule parameters
        self.update_every = update_every
        self.start_update_after_step = start_update_after_step
        if beta is not None:
            self.target_beta = beta
            self._beta = self._classical_beta
            self.inverse_gamma = inverse_gamma
            self.power = power
        elif sigma_rel is not None:
            self.gamma = ema_length_to_gamma(sigma_rel)
            self._beta = self._karras_beta
        else:
            raise ValueError("Either beta or sigma_rel must be provided.")
        
        # Initialize models
        self.online_model = model             # Hare
        self.ema_model = copy.deepcopy(model) # Tortoise
        
        # Freeze Tortoise 
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        for buffer in self.ema_model.buffers(): # e.g. GroupNorm's running_mean
            buffer.requires_grad_(False)
    
    @property
    def beta(self):
        """ EMA decay parameter."""
        return self._beta()
    
    def _karras_beta(self):
        """ Karras EMA. """
        dt = self.update_every
        t = self.step
        return (1 - dt/t) ** (self.gamma + 1)
    
    def _classical_beta(self):
        """ Classical EMA with warmup schedule."""
        step = max(0, self.step - self.start_update_after_step)
        value = 1 - (1 + self.step / self.inverse_gamma) ** (-self.power)
        return min(max(value, 0.5), self.target_beta)

    def update(self):
        """
        Update the EMA model with the online model.
        """
        self.step += 1
        if self.step % self.update_every == 0 and self.step > self.start_update_after_step:
            with torch.no_grad():
                # Update parameters
                for ema_param, online_param in zip(self.ema_model.parameters(), self.online_model.parameters()):
                    ema_param.copy_(self.beta * ema_param + (1 - self.beta) * online_param)
                
                # Update buffers
                for ema_buffer, online_buffer in zip(self.ema_model.buffers(), self.online_model.buffers()):
                    ema_buffer.copy_(online_buffer)
    
    def soft_reset(self):
        """
        Reinitialize the online model with the EMA model parameters.
        """
        with torch.no_grad():
            for online_param, ema_param in zip(self.online_model.parameters(), self.ema_model.parameters()):
                online_param.copy_(ema_param)
    
