from functools import partial
import warnings
import torch
import torch.nn as nn
import os, json, re
from glob import glob
import numpy as np
from torch.nn import Module
from typing import Union

DTYPE = torch.float32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | cond_batch | cond_instance | cond_instance++ | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = nn.Identity
    elif norm_type == "cond_batch":
        from .layers.conditional_batchnorm2d import ConditionalBatchNorm2d
        norm_layer = ConditionalBatchNorm2d
    elif norm_type == "cond_instance":
        from .layers.conditional_instancenorm2d import ConditionalInstanceNorm2d
        norm_layer = ConditionalInstanceNorm2d
    elif norm_type == "cond_instance++":
        from .layers.conditional_instancenorm2d_plus import ConditionalInstanceNorm2dPlus
        norm_layer = ConditionalInstanceNorm2dPlus
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_activation(activation_type="elu"):
    if activation_type is None:
        return nn.Identity()
    elif activation_type.lower() == "none":
        return nn.Identity()
    elif activation_type == "relu":
        activation = nn.ReLU()
    elif activation_type == "elu":
        activation = nn.ELU()
    elif activation_type == "tanh":
        activation = nn.Tanh()
    elif activation_type in ["swish", "silu"]:
        activation = nn.SiLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return activation


def load_architecture(
        checkpoints_directory, 
        model: Union[str, Module] = None, 
        dimensions=2, 
        hyperparameters=None, 
        device=DEVICE,
        model_checkpoint:int=None,
        ) -> list[Module, dict]:
    if hyperparameters is None:
        hyperparameters = {}
    if model is None:
        with open(os.path.join(checkpoints_directory, "model_hparams.json"), "r") as f:
            hparams = json.load(f)
        hyperparameters.update(hparams)
        model = hparams.get("model_architecture", "ncsnpp")
        if "dimensions" not in hyperparameters.keys():
            hyperparameters["dimensions"] = dimensions
    if isinstance(model, str):
        if model.lower() == "ncsnpp":
            from score_models.architectures import NCSNpp
            model = NCSNpp(**hyperparameters).to(device)
        elif model.lower() == "ddpm":
            from score_models.architectures import DDPM
            model = DDPM(**hyperparameters).to(device)
        elif model.lower() == "mlp":
            from score_models import MLP
            model = MLP(**hyperparameters).to(device)
        else:
            raise ValueError(f"{model} not supported")
    # Backward compatibility with old stuff
    if "sde" in hyperparameters.keys():
        if hyperparameters["sde"] == "vpsde":
            hyperparameters["sde"] = "vp"
        elif hyperparameters["sde"] == "vesde":
            hyperparameters["sde"] = "ve"
    if checkpoints_directory is not None:
        paths = glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
        checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
        if not paths:
            warnings.warn(f"Directory {checkpoints_directory} might not have checkpoint files. Cannot load architecture.")
            return model, hyperparameters, None
        if model_checkpoint is None:
            checkpoint = np.argmax(checkpoints)
            path = paths[checkpoint]
        elif model_checkpoint not in checkpoints:
            warnings.warn(f"Directory {checkpoints_directory} does not have the checkpoint requested. Methods defaults to loading latest checkpoint.")
            checkpoint = np.argmax(checkpoints)
            path = paths[checkpoint]
        else:
            checkpoint = [i for i, c in enumerate(checkpoints) if c == model_checkpoint][0]
            path = paths[checkpoint]
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model_dir = os.path.split(checkpoints_directory)[-1]
            print(f"Loaded checkpoint {checkpoints[checkpoint]} of {model_dir}")
        except (KeyError, RuntimeError):
            # Maybe the ScoreModel instance was used when saving the weights, in which case we hack the loading process
            from score_models import ScoreModel
            model = ScoreModel(model, **hyperparameters)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.model # Remove the ScoreModel wrapping to extract the nn
            model_dir = os.path.split(checkpoints_directory)[-1]
            print(f"Loaded checkpoint {checkpoints[checkpoint]} of {model_dir}")
        return model, hyperparameters, checkpoints[checkpoint]
    return model, hyperparameters, None

