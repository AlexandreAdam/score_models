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
        self.use_num_updates = use_num_updates
        self.num_updates = 0
        self.shadow_params = eqx.tree_at(lambda _: True, model, lambda x: copy.deepcopy(x))

    def update(self, model: eqx.Module):
        decay = self.decay
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        
        def ema_update(shadow_param, model_param):
            return shadow_param * decay + model_param * one_minus_decay
        
        model_params = eqx.tree_at(lambda _: True, model, lambda x: x)
        self.shadow_params = jax.tree_map(ema_update, self.shadow_params, model_params)

    @contextlib.contextmanager
    def apply_averaging(self, model: eqx.Module):
        original_params = eqx.tree_at(lambda _: True, model, lambda x: copy.deepcopy(x))
        try:
            self.copy_to(model)
            yield model
        finally:
            self.restore(model, original_params)

    def copy_to(self, model: eqx.Module):
        def copy_fn(_, shadow_param):
            return copy.deepcopy(shadow_param)
        new_model = eqx.tree_at(lambda _: True, model, copy_fn, self.shadow_params)
        update_model_params(model, model_state_dict(new_model))


    def restore(self, model: eqx.Module, original_params):
        if original_params is not None:
            update_model_params(model, self.original_params)
        else:
            raise RuntimeError("No parameters have been stored.")

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]

