import jax.numpy as jnp
import flax.linen as nn
import functools
import jax
import os
import json
import re
import numpy as np
import equinox as eqx
from glob import glob
from jax import lax
from jaxtyping import PRNGKeyArray
from typing import Optional


def stop_gradient(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return lax.stop_gradient(result)
    return wrapper


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer for JAX/Flax.
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    """
    if norm_type == "batch":
        norm_layer = nn.BatchNorm
    elif norm_type == "instance":
        norm_layer = (
            nn.GroupNorm
        )  # Note: Flax doesn't have a direct InstanceNorm, using GroupNorm as an approximation
    elif norm_type == "none":

        def identity_layer(x):
            return x

        norm_layer = identity_layer
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return norm_layer


def get_activation(activation_type: Optional[str]):
    """Return a JAX activation function."""
    if activation_type is None or activation_type.lower() == "none":
        return lambda x: x
    elif activation_type == "relu":
        return jax.nn.relu
    elif activation_type == "elu":
        return jax.nn.elu
    elif activation_type == "tanh":
        return jnp.tanh
    elif activation_type in ["swish", "silu"]:
        return jax.nn.silu
    else:
        raise NotImplementedError(f"activation layer [{activation_type}] is not found")


def torch_state_to_jax(torch_state_dict):
    """Convert PyTorch state dict to a format suitable for JAX/Equinox models."""
    jax_state_dict = {}
    for k, v in torch_state_dict.items():
        jax_state_dict[k] = jnp.array(v.cpu().numpy())
    return jax_state_dict


def load_state_dict_from_pt(path: str) -> dict:
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required to load PyTorch checkpoints." 
                          " You can install the cpu version of torch to avoid conflicting cuda dependencies."
                          " Instructions to install the cpu version can be found here https://pytorch.org/get-started/locally/")
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    state_dict = torch_state_to_jax(state_dict)
    return state_dict


def update_model_params(model, state_dict):
    def update(module, name):
        for key, value in module.__dict__.items():
            if isinstance(value, eqx.Module):
                update(value, f"{name}.{key}")
            elif key in state_dict and name:
                setattr(module, key, state_dict[f"{name}.{key}"])
            elif key in state_dict:
                setattr(module, key, state_dict[key])
    update(model, "")
    return model


def model_state_dict(model, prefix=''):
    state_dict = {}
    for key, value in model.__dict__.items():
        if isinstance(value, eqx.Module):
            state_dict.update(model_state_dict(value, prefix=f"{prefix}{key}."))  # Recursive call for nested modules
    return state_dict


def load_model_from_pt(path: str, model: nn.Module) -> nn.Module:
    state_dict = load_state_dict_from_pt(path)
    model = update_model_params(model, state_dict)
    return model


def load_architecture(
        checkpoints_directory, 
        model=None, 
        model_checkpoint=None, 
        hyperparameters=None,
        key: Optional[PRNGKeyArray] = None
        ):
    if hyperparameters is None:
        hyperparameters = {}
    if model is None:
        with open(os.path.join(checkpoints_directory, "model_hparams.json"), "r") as f:
            hparams = json.load(f)
        hyperparameters.update(hparams)
        model_architecture = hparams.get("model_architecture", "ncsnpp")
        if key is None:
            key = jax.random.PRNGKey(0)
        
        if model_architecture.lower() == "ncsnpp":
            from score_models.jax.architectures import NCSNpp
            model = NCSNpp(**hyperparameters, key=key)
            
        elif model_architecture.lower() == "ddpm":
            from score_models.jax.architectures import DDPM
            model = DDPM(**hyperparameters, key=key)
            
        elif model_architecture.lower() == "mlp":
            from score_models.jax.architectures import MLP
            model = MLP(**hyperparameters, key=key)
        else:
            raise ValueError(f"{model_architecture} not supported")
    else:
        # Directly use the provided model if not a string
        pass

    if "sde" in hyperparameters:
        if hyperparameters["sde"] == "vesde":
            hyperparameters["sde"] = "vp"
        elif hyperparameters["sde"] == "vesde":
            hyperparameters["sde"] = "ve"

    if checkpoints_directory is not None:
        paths_pt = glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
        paths_pkl = glob(os.path.join(checkpoints_directory, "checkpoint*.eqx"))
        
        if len(paths_pt) > 0 and len(paths_pkl) > 0:
            print("Found pt and pkl File. Loading the eqx files.")
        if len(paths_pkl) > 0:
            paths = paths_pkl
            extension = "eqx"
        elif len(paths_pt) > 0:
            paths = paths_pt
            extension = "pt"
        else:
            print(f"Directory {checkpoints_directory} might not have checkpoint files. Cannot load architecture.")
            return model, hyperparameters, None

        checkpoints = [int(re.findall(r"\d+", os.path.basename(path))[-1]) for path in paths]
        if model_checkpoint is None:
            checkpoint = np.argmax(checkpoints)
            path = paths[checkpoint]
        else:
            checkpoint = model_checkpoint
            path = os.path.join(checkpoints_directory, f"checkpoint_{checkpoint}.{extension}")
            if not os.path.exists(path):
                print(f"Checkpoint {model_checkpoint} not found. Loading latest checkpoint.")
                checkpoint = np.argmax(checkpoints)
                path = paths[checkpoint]
        
        if extension == "pt":
            try:
                model = load_model_from_pt(path, model)
            except (KeyError, RuntimeError):
                # Maybe the ScoreModel instance was used when saving the weights, in which case we hack the loading process
                from score_models import ScoreModel
                model = ScoreModel(model, **hyperparameters)
                model = load_model_from_pt(path, model)
        else: # eqx
            with open(path, "rb") as f:
                model = eqx.tree_deserialise_leaves(f, model)
            
        model_dir = os.path.split(checkpoints_directory)[-1]
        print(f"Loaded checkpoint {checkpoints[checkpoint]} of {model_dir}")
        return model, hyperparameters, checkpoint

    return model, hyperparameters, None

