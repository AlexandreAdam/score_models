import jax.numpy as jnp
import flax.linen as nn
import jax
import os
import json
from glob import glob
import re
import numpy as np
from typing import Union, Callable, Optional


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


def load_architecture(
    checkpoints_directory, model=None, dimensions=2, hyperparameters=None
):
    if hyperparameters is None:
        hyperparameters = {}
    if model is None:
        with open(os.path.join(checkpoints_directory, "model_hparams.json"), "r") as f:
            hparams = json.load(f)
        hyperparameters.update(hparams)
        model_architecture = hparams.get("model_architecture", "ncsnpp")
        if model_architecture.lower() == "ncsnpp":
            from score_models.architectures import NCSNpp

            model = NCSNpp(**hyperparameters)
        elif model_architecture.lower() == "ddpm":
            from score_models.architectures import DDPM

            model = DDPM(**hyperparameters)
        elif model_architecture.lower() == "mlp":
            from score_models.architectures import MLP

            model = MLP(**hyperparameters)
        else:
            raise ValueError(f"{model_architecture} not supported")
    else:
        # Directly use the provided model if not a string
        pass

    if "sde" in hyperparameters:
        if hyperparameters["sde"] == "vpsde":
            hyperparameters["sde"] = "vp"
        elif hyperparameters["sde"] == "vesde":
            hyperparameters["sde"] = "ve"

    if checkpoints_directory is not None:
        paths = glob(
            os.path.join(checkpoints_directory, "checkpoint*.pt")
        )
        checkpoints = [
            int(re.findall(r"\d+", os.path.basename(path))[-1]) for path in paths
        ]
        if not paths:
            print(
                f"Directory {checkpoints_directory} might not have checkpoint files. Cannot load architecture."
            )
            return model, hyperparameters, None
        if model_checkpoint is None:
            checkpoint = np.argmax(checkpoints)
            path = paths[checkpoint]
        else:
            path = os.path.join(
                checkpoints_directory, f"checkpoint_{model_checkpoint}.pt"
            )  # Adjust file naming
            if not os.path.exists(path):
                print(
                    f"Checkpoint {model_checkpoint} not found. Loading latest checkpoint."
                )
                checkpoint = np.argmax(checkpoints)
                path = paths[checkpoint]

        with open(path, "rb") as f:
            bytes_input = f.read()
            params = serialization.from_bytes(
                model.init(jax.random.PRNGKey(0), jnp.ones((1, dimensions))),
                bytes_input,
            )
            model = model.bind(params)
            print(f"Loaded checkpoint from {path}")

    return model, hyperparameters, checkpoint
