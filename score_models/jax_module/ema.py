"""
Translated from
https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py
to match the same behavior as in the torch_module fitting method.
"""
import jax
import copy
import contextlib
import equinox as eqx
from .utils import update_model_params, model_state_dict


class ExponentialMovingAverage:
    def __init__(self, model: eqx.Module, decay: float, use_num_updates: bool = True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.model = model
        self.use_num_updates = use_num_updates
        self.num_updates = 0
        self.shadow_params = eqx.tree_at(lambda _: True, model, replace=copy.deepcopy)
        self.original_params = None

    def update(self):
        decay = self.decay
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        
        def ema_update(shadow_param, model_param):
            return shadow_param * decay + model_param * one_minus_decay
        
        self.shadow_params = jax.tree_map(ema_update, self.shadow_params, model_state_dict(self.model))

    def copy_to(self):
        def copy_fn(_, shadow_param):
            return copy.deepcopy(shadow_param)
        new_model = eqx.tree_at(lambda _: True, self.model, copy_fn, self.shadow_params)
        update_model_params(self.model, model_state_dict(new_model))

    def store(self):
        self.original_params = eqx.tree_at(lambda _: True, self.model, replace=copy.deepcopy)

    def restore(self):
        if self.original_params is not None:
            update_model_params(self.model, self.original_params)
        else:
            raise RuntimeError("No parameters have been stored.")

    @contextlib.contextmanager
    def average_parameters(self):
        self.store()
        self.copy_to()
        try:
            yield
        finally:
            self.restore()

    def to(self, device=None, dtype=None):
        # This method is specific to PyTorch and may not have a direct equivalent in JAX/Equinox
        pass

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "original_params": self.original_params,
        }

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]
        self.original_params = state_dict["original_params"]

