from typing import Optional, Literal, Union

import torch
import torch.nn as nn
from torch.nn import Module
from ..layers import GaussianFourierProjection, PositionalEncoding


def validate_conditional_arguments(
        conditions: Optional[tuple[Literal["time_discrete", "time_continuous", "time_vector", "input_tensor"]]] = None,
        condition_embeddings:  Optional[tuple[int]] = None,
        condition_channels: Optional[tuple[int]] = None
        ):
    discrete_index = 0 # Number of discrete conditional variables
    tensor_index = 0 # Number of vector/tensor conditional variables
    if conditions:
        if not isinstance(conditions, (tuple, list)):
            raise ValueError("Conditions should be a tuple of strings.")
        for c in conditions:
            if c.lower() not in ["time_discrete", "time_continuous", "time_vector", "input_tensor"]:
                raise ValueError(f"Conditions must be either 'discrete', 'continuous', 'vector', 'tensor'], received {c}.")
            if c.lower() == "time_discrete":
                if condition_embeddings is None:
                    raise ValueError("condition_embeddings must be provided for a 'discrete' condition type, "
                                     "and must be a tuple of integers of length equal to the number of 'discrete' conditions.")
                if len(condition_embeddings) <= discrete_index:
                    raise ValueError("condition_embeddings must be provided for a 'discrete' condition type, " 
                                     "and must be a tuple of integers of length equal to the number of 'discrete' conditions.")
                if not isinstance(condition_embeddings, (tuple, list)) or not isinstance(condition_embeddings[discrete_index], int):
                    raise ValueError("condition_embeddings must be provided and be a tuple of integers for a 'discrete' condition type")
                discrete_index += 1
            elif c.lower() in ["input_tensor", "time_vector"]:
                if condition_channels is None:
                    raise ValueError("condition_channels must be provided for 'input_tensor' and 'time_vector' condition types, "
                                     "and must be a tuple of integers of length equal to the number of 'input_tensor' and 'time_vector' conditions.")
                if len(condition_channels) <= tensor_index:
                    raise ValueError("condition_channels must be provided for 'input_tensor' and 'time_vector' condition types, "
                                     "and must be a tuple of integers of length equal to the number of 'input_tensor' and 'time_vector' conditions.")
                if not isinstance(condition_channels, (tuple, list)) or not isinstance(condition_channels[tensor_index], int):
                    raise ValueError("condition_channels must be provided and be a tuple of integers for 'input_tensor' and 'time_vector' condition type.")
                tensor_index += 1


def conditional_branch(
        model: Module,
        time_branch_channels: int,
        input_branch_channels: int,
        condition_embeddings: Union[tuple[int], type(None)],
        condition_channels: Union[tuple[int], type(None)],
        fourier_scale: float = 30.,
        ):
    total_time_channels = time_branch_channels
    total_input_channels = input_branch_channels
    conditional_branch = []
    for condition_type in model.condition_type:
        if condition_type.lower() == "time_discrete":
            conditional_branch.append(
                    nn.Embedding(
                        num_embeddings=condition_embeddings[0],
                        embedding_dim=time_branch_channels
                        )
                    )
            condition_embeddings = condition_embeddings[1:]
            total_time_channels += time_branch_channels 
        
        elif condition_type.lower() == "time_continuous":
            conditional_branch.append(
                    GaussianFourierProjection(
                        embed_dim=time_branch_channels, 
                        scale=fourier_scale
                        )
                    )
            total_time_channels += time_branch_channels
        
        elif condition_type.lower() == "time_vector":
            conditional_branch.append(
                    PositionalEncoding(
                        channels=condition_channels[0],
                        embed_dim=time_branch_channels,
                        scale=fourier_scale
                        )
                    )
            condition_channels = condition_channels[1:]
            total_time_channels += time_branch_channels
        
        elif condition_type.lower() == "input_tensor":
            total_input_channels += condition_channels[0]
            condition_channels = condition_channels[1:]
            
    model.conditional_branch = nn.ModuleList(conditional_branch)
    return total_time_channels, total_input_channels


def merge_conditional_time_branch(model, temb, *args):
    B, *_ = temb.shape
    c_idx = 0
    e_idx = 0
    if len(args) != len(model.condition_type):
        raise ValueError(f"The network requires {len(model.condition_type)} additional arguments, but {len(args)} were provided.")
    for condition, condition_type in zip(args, model.condition_type):
        if "time" in condition_type.lower():
            if "discrete" in condition_type.lower():
                if torch.any((condition < 0) | (condition >= model.condition_embeddings[e_idx])):
                    raise ValueError(f"Additional argument {c_idx} must be a long tensor with values "
                                      f"between 0 and {model.condition_embeddings[e_idx]-1} inclusively.")
                e_idx += 1
            c_emb = model.conditional_branch[c_idx](condition).view(B, -1)
            temb = torch.cat([temb, c_emb], dim=1)
            c_idx += 1
    return temb

def merge_conditional_input_branch(model, x, *args):
    B, *D = x.shape
    for condition, condition_type in zip(args, model.condition_type):
        if "input" in condition_type.lower():
            x = torch.cat([x, condition], dim=1)
    return x
