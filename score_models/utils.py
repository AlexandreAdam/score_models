from functools import partial
import torch
import torch.nn as nn
import os, json, re
from glob import glob
import numpy as np
from .definitions import DEVICE


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
    if activation_type == "relu":
        activation = nn.ReLU()
    elif activation_type == "elu":
        activation = nn.ELU()
    elif activation_type == "tanh":
        activation = nn.Tanh()
    elif activation_type == "swish":
        activation = nn.SiLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return activation


def load_model(checkpoints_directory, model: str, eval=True):
    with open(os.path.join(checkpoints_directory, "model_hparams.json"), "r") as f:
        hyperparameters = json.load(f)
    if model.lower() == "ncsnpp1d":
        from score_models import NCSNpp1d
        model = NCSNpp1d(**hyperparameters).to(DEVICE)
    elif model.lower() in ["ncsnpp", "ncsnpp2d"]:
        from score_models import NCSNpp
        model = NCSNpp(**hyperparameters).to(DEVICE)
    elif model.lower() == "ncsnpp3d":
        from score_models import NCSNpp3d
        model = NCSNpp3d(**hyperparameters).to(DEVICE)
    elif model.lower() == "ddpm":
        from score_models import DDPM
        model = DDPM(**hyperparameters).to(DEVICE)
    paths = glob(os.path.join(checkpoints_directory, "*.pt"))
    checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
    model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
    if eval:
        model.eval()
    return model
