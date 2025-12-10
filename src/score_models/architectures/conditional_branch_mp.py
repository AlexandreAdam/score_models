from typing import Optional, Literal, Union

import torch
import torch.nn as nn
from torch.nn import Module
from .conditional_branch import validate_conditional_arguments
from ..layers import MPFourier, MPConv, mp_cat, MPPositionalEncoding


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
                    MPFourier(
                        width=time_branch_channels, 
                        bandwidth=fourier_scale
                        )
                    )
            total_time_channels += time_branch_channels
        
        elif condition_type.lower() == "time_vector":
            conditional_branch.append(
                    MPPositionalEncoding(
                        channels=condition_channels[0],
                        width=time_branch_channels,
                        bandwidth=fourier_scale
                        )
                    )
            condition_channels = condition_channels[1:]
            total_time_channels += time_branch_channels
        
        elif condition_type.lower() == "input_tensor":
            total_input_channels += condition_channels[0]
            condition_channels = condition_channels[1:]
            
    model.conditional_branch = nn.ModuleList(conditional_branch)
    return total_time_channels, total_input_channels


def merge_conditional_time_branch(model, temb, *args, condition_balance=0.5):
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
            temb = mp_cat(temb, c_emb, dim=1, t=condition_balance) # Magnitude preserving concatenation with control over the balance
            c_idx += 1
    return temb

def merge_conditional_input_branch(model, x, *args, condition_balance=0.5):
    B, *D = x.shape
    for condition, condition_type in zip(args, model.condition_type):
        if "input" in condition_type.lower():
            x = mp_cat(x, condition, dim=1, t=condition_balance) # Magnitude preserving concatenation with control over the balance
    return x
