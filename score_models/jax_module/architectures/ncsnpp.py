# import jax.numpy as jnp
# import functools
# from jax import random
# import equinox as eqx
# from functools import partial
# from typing import List, Optional, Tuple


# class NCSNpp(eqx.Module):
    # modules: List[eqx.Module]
    # act: Callable
    # attention: bool
    # conditioned: bool
    # condition_type: Tuple[str, ...]
    # condition_embedding_layers: List[eqx.Module]
    # all_modules: eqx.ModuleList

    # def __init__(self,
                 # channels: int = 1,
                 # dimensions: int = 2,
                 # nf: int = 128,
                 # ch_mult: Tuple[int, ...] = (2, 2, 2, 2),
                 # num_res_blocks: int = 2,
                 # activation_type: str = "swish",
                 # dropout: float = 0.,
                 # resample_with_conv: bool = True,
                 # fir: bool = True,
                 # fir_kernel: Tuple[int, ...] = (1, 3, 3, 1),
                 # skip_rescale: bool = True,
                 # progressive: str = "output_skip",
                 # progressive_input: str = "input_skip",
                 # init_scale: float = 1e-2,
                 # fourier_scale: float = 16.,
                 # resblock_type: str = "biggan",
                 # combine_method: str = "sum",
                 # attention: bool = True,
                 # condition: Optional[Tuple[str, ...]] = ("None",),
                 # condition_num_embedding: Optional[Tuple[int, ...]] = None,
                 # condition_input_channels: Optional[int] = None,
                 # condition_vector_channels: Optional[int] = None,
                 # **kwargs):
        # super().__init__()
        # if dimensions not in [1, 2, 3]:
            # raise ValueError("Input must have 1, 2, or 3 spatial dimensions to use this architecture")
        # self.conditioned = False
        # discrete_index = 0
        # if condition is not None:
            # if not isinstance(condition, (tuple, list)):
                # raise ValueError("Condition should be a list or a tuple of strings")
        # for c in condition:
            # if c.lower() not in ["none", "discrete_timelike", "continuous_timelike", "vector", "input"]:
                # raise ValueError(f"Condition must be in ['none', 'discrete_timelike', 'continuous_timelike', 'input'], received {c}")
            # if c.lower() != "none":
                # self.conditioned = True
            # elif c.lower() == "none" and self.conditioned:
                # raise ValueError(f"Cannot have a mix of 'None' and other type of conditions, received the tuple {condition}")
            # if c.lower() == "discrete_timelike":
                # if not isinstance(condition_num_embedding, (tuple, list)):
                    # raise ValueError("condition_num_embedding must be provided and be a tuple or list of integer for discrete_timelike condition type")
                # elif not isinstance(condition_num_embedding[discrete_index], int):
                    # raise ValueError("condition_num_embedding must be provided and be a tuple or list of integer for discrete_timelike condition type")
                # discrete_index += 1
            # elif c.lower() == "input":
                # if not isinstance(condition_input_channels, int):
                    # raise ValueError("condition_input_channels must be provided and be an integer for input condition type")
            # elif c.lower() == "vector":
                # if not isinstance(condition_vector_channels, int):
                    # raise ValueError("condition_vector_channels must be provided and be an integer for vector condition type")

        # self.condition_type = condition
        # self.condition_num_embedding = condition_num_embedding
        # self.condition_input_channels = 0 if condition_input_channels is None else condition_input_channels
        # self.condition_vector_channels = condition_vector_channels
        # self.dimensions = dimensions
        # self.channels = channels
        # self.hyperparameters = {
            # "channels": channels,
            # "nf": nf,
            # "activation_type": activation_type,
            # "ch_mult": ch_mult,
            # "num_res_blocks": num_res_blocks,
            # "resample_with_conv": resample_with_conv,
            # "dropout": dropout,
            # "fir": fir,
            # "fir_kernel": fir_kernel,
            # "skip_rescale": skip_rescale,
            # "progressive": progressive,
            # "progressive_input": progressive_input,
            # "init_scale": init_scale,
            # "fourier_scale": fourier_scale,
            # "resblock_type": resblock_type,
            # "combine_method": combine_method,
            # "attention": attention,
            # "dimensions": dimensions,
            # "condition": condition,
            # "condition_num_embedding": condition_num_embedding,
            # "condition_input_channels": condition_input_channels,
            # "condition_vector_channels": condition_vector_channels
        # }
        # self.act = act = get_activation(activation_type)
        # self.attention = attention

        # self.nf = nf
        # self.num_res_blocks = num_res_blocks
        # self.num_resolutions = num_resolutions = len(ch_mult)

        # self.skip_rescale = skip_rescale
        # self.progressive = progressive.lower()
        # self.progressive_input = progressive_input.lower()
        # self.resblock_type = resblock_type
        # assert progressive in ['none', 'output_skip', 'residual']
        # assert progressive_input in ['none', 'input_skip', 'residual']
        # combiner = functools.partial(Combine, method=combine_method.lower(), dimensions=self.dimensions)
 

        # condition_embedding_layers = []
        # if condition is not None:
            # for c in condition:
                # if c.lower() == "discrete_timelike":
                    # # Assuming an embedding layer for discrete conditions
                    # key, subkey = random.split(key)
                    # condition_embedding_layers.append(eqx.nn.Embedding(num_embeddings=condition_num_embedding[discrete_index], embedding_dim=nf, key=subkey))
                    # discrete_index += 1
                # elif c.lower() == "continuous_timelike":
                    # # Assuming GaussianFourierProjection for continuous conditions
                    # key, subkey = random.split(key)
                    # condition_embedding_layers.append(GaussianFourierProjection(embed_dim=nf, scale=fourier_scale, key=subkey))
                # elif c.lower() == "vector":
                    # # Assuming PositionalEncoding for vector conditions
                    # key, subkey = random.split(key)
                    # condition_embedding_layers.append(PositionalEncoding(channels=condition_vector_channels, embed_dim=nf, scale=fourier_scale, key=subkey))

        # self.condition_embedding_layers = eqx.ModuleList(condition_embedding_layers)

        # # Model building
        # modules = []
        # # Add initial time embedding module
        # key, subkey = random.split(key)
        # modules.append(GaussianFourierProjection(embed_dim=nf, scale=fourier_scale, key=subkey))

        # # Continue adding other modules as in the original implementation...

        # self.all_modules = eqx.ModuleList(modules)
        # self.act = get_activation(activation_type)
        # self.attention = attention
        
        
        # AttnBlock = SelfAttentionBlock
        # Upsample = partial(UpsampleLayer, with_conv=resample_with_conv, fir=fir, 
                           # fir_kernel=fir_kernel, dimensions=dimensions)
        # Downsample = partial(DownsampleLayer, with_conv=resample_with_conv, fir=fir, 
                             # fir_kernel=fir_kernel, dimensions=dimensions)

        # ResnetBlock = partial(ResnetBlockBigGANpp, act=self.act, temb_dim=4 * nf,
                              # dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                              # init_scale=init_scale, skip_rescale=skip_rescale,
                              # dimensions=dimensions)

        # modules = []
        # # Time embedding layers
        # modules += [GaussianFourierProjection(embed_dim=nf, scale=fourier_scale), 
                    # eqx.nn.Linear(nf, nf * 4), eqx.nn.Linear(nf * 4, nf * 4)]

        # # Initial conv layer
        # modules.append(conv3x3(channels, nf, dimensions=dimensions))
        # in_ch = nf
        # hs_c = [nf]
        
        # # Downsample layers
        # for i_level in range(self.num_resolutions):
            # out_ch = nf * ch_mult[i_level]
            # for i_block in range(num_res_blocks):
                # modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                # in_ch = out_ch
                # hs_c.append(in_ch)
            # if i_level != self.num_resolutions - 1:
                # modules.append(Downsample(in_ch=in_ch))
                # hs_c.append(in_ch)

        # # Middle layers
        # modules.append(ResnetBlock(in_ch=in_ch))
        # if self.attention:
            # modules.append(AttnBlock(channels=in_ch, dimensions=dimensions))
        # modules.append(ResnetBlock(in_ch=in_ch))

        # # Upsample layers
        # for i_level in reversed(range(self.num_resolutions)):
            # out_ch = nf * ch_mult[i_level]
            # for i_block in range(num_res_blocks + 1):
                # modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                # in_ch = out_ch
            # if i_level != 0:
                # modules.append(Upsample(in_ch=in_ch))

        # modules.append(eqx.nn.GroupNorm(num_channels=in_ch, num_groups=min(in_ch // 4, 32)))
        # modules.append(conv3x3(in_ch, channels, dimensions=dimensions))
        # self.modules = modules


    # def __call__(self, t, x, *condition_args, key=None):
        # # Implementation of the forward pass, handling conditions, etc.
        # pass

