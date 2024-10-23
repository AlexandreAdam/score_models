# from typing import Optional

# import torch
# from torch.nn import Module
# from torch import nn

# from .ncsnpp import NCSNpp

# class NCSNppLevel(Module):
    # def __init__(
            # self, 
            # base_net: NCSNpp,
            # ):
        # super().__init__()







# def adjust_layer_index(name, shift):
    # """
    # Adjusts the index in layer names to account for the extra levels added.
    # """
    # match = re.search(r'all_modules\.(\d+)', name)
    # if match:
        # original_index = int(match.group(1))
        # # Adjust index for layers beyond the initial Fourier and Linear layers
        # if original_index > 3:  # Assuming the first few layers are fixed as described
            # return original_index + shift
    # else:
        # raise ValueError(f"Layer index not found in layer name: {name}")


# def initialize_from_pretrained(pretrained_score_model, extra_levels=1, ch_mult=1) -> ScoreModel:
    # """
    # Initializes a new model with additional levels from a pre-trained bottleneck model.
    
    # Args:
        # pretrained_model (NCSNpp): The pre-trained model from which to initialize the new model.
        # extra_levels (int): The number of additional levels to add to the U-Net architecture.
        # ch_mult (int): The channel multiplier for the additional levels. If set to 1, the first layer of the bottleneck
                       # is initialized with pre-trained weights; otherwise, it is initialized with random weights.
    
    # Returns:
        # NCSNpp: A new model instance with the updated architecture and weights.
    # """
    # score_hyperparameters = pretrained_score_model.hyperparameters.copy()
    # hyperparameters = pretrained_score_model.model.hyperparameters.copy()
    
    # # Adjust ch_mult for the new model
    # hyperparameters["ch_mult"] = [ch_mult] * extra_levels + hyperparameters["ch_mult"]
    
    # # Initialize the new model
    # new_model = NCSNpp(**hyperparameters)

    # pretrained_dict = pretrained_score_model.model.state_dict()
    # new_dict = new_model.state_dict()
    
    # # Layer index shift calculation
    # num_res_blocks = hyperparameters["num_res_blocks"]
    # input_skip_or_residual = hyperparameters["progressive_input"] in ["input_skip", "residual"]
    # layer_shift = (num_res_blocks + (2 if input_skip_or_residual else 1)) * extra_levels
    
    # # Copy weights with adjustments
    # for name, param in pretrained_dict.items():
        # # Adjust layer names based on the calculated shift
        # if ch_mult != 1 and "all_modules.3.conv" in name:
            # adjusted_index = adjust_layer_index(name, layer_shift)
            # new_name = name.replace(f".{name.split('.')[1]}.", f".{adjusted_index}.")
            
            # if new_name in new_dict:
                # new_dict[new_name].copy_(param)
                # print(f"Copied weights for layer: {name} -> {new_name}")
            # else:
                # print(f"Layer {new_name} not found in new model. This layer might be part of the added level.")
    
    # # Handle special case for the first layer of the bottleneck if ch_mult != 1
    # if ch_mult != 1:
        # # Initialize first layer of bottleneck with random weights or according to some strategy
        # print("Initializing first layer of the bottleneck with random weights due to ch_mult != 1.")
    
    # new_score_model = ScoreModel(new_model, **score_hyperparameters)
    # return new_score_model

