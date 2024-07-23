from typing import Optional, Literal

import torch


def validate_conditional_arguments(
        conditions: Optional[tuple[Literal["discrete", "continuous", "vector", "tensor"]]] = None,
        condition_embeddings:  Optional[tuple[int]] = None,
        condition_channels: Optional[tuple[int]] = None
        ):
    discrete_index = 0 # Number of discrete conditional variables
    tensor_index = 0 # Number of vector/tensor conditional variables
    if conditions:
        if not isinstance(conditions, tuple):
            raise ValueError("Conditions should be a tuple of strings.")
        for c in conditions:
            if c.lower() not in ["discrete", "continuous", "vector", "tensor"]:
                raise ValueError(f"Conditions must be either 'discrete', 'continuous', 'vector', 'tensor'], received {c}.")
            if c.lower() == "discrete":
                if not isinstance(condition_embeddings, tuple) or not isinstance(condition_embeddings[discrete_index], int):
                    raise ValueError("condition_embeddings must be provided and be a tuple of integers for a 'discrete' condition type")
                discrete_index += 1
            elif c.lower() in ["tensor", "vector"]:
                if not isinstance(condition_channels, tuple) or not isinstance(condition_channels[tensor_index], int):
                    raise ValueError("condition_channels must be provided and be a tuple of integers for 'tensor' condition type.")
                tensor_index += 1
