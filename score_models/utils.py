from .layers.conditional_batchnorm2d import ConditionalBatchNorm2d
from .layers.conditional_instancenorm2d_plus import ConditionalInstanceNorm2dPlus
from .layers.conditional_instancenorm2d import ConditionalInstanceNorm2d
from functools import partial
import torch.nn as nn


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
        norm_layer = ConditionalBatchNorm2d
    elif norm_type == "cond_instance":
        norm_layer = ConditionalInstanceNorm2d
    elif norm_type == "cond_instance++":
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
