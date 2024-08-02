from typing import Callable, Literal, List

import torch
from torch import Tensor
from torch.func import vjp

__all__ = ['divergence_with_hutchinson_trick']


def divergence_with_hutchinson_trick(
        drift: Callable[[Tensor, Tensor, List[Tensor]], Tensor],
        t: Tensor, 
        x: Tensor, 
        *args, 
        cotangent_vectors: int = 1, 
        noise_type: Literal['rademacher', 'gaussian'] = 'rademacher',
        **kwargs
        ) -> Tensor:
    """
    Compute the divergence of the drift function using the Hutchinson trace estimator.
    
    Args:
        drift: Drift function of the ODE.
        t: Time of the ODE.
        x: State of the ODE.
        *args: Additional arguments to pass to the drift function.
        cotangent_vectors: Number of cotangent vectors to sample for the Hutchinson trace estimator.
        noise_type: Type of noise to sample, either 'rademacher' or 'gaussian'.
    """
    B, *D = x.shape
    # duplicate samples for for the Hutchinson trace estimator
    samples = torch.tile(x, [cotangent_vectors, *[1]*len(D)])
    t = torch.tile(t, [cotangent_vectors])
    _args = []
    for arg in args:
        _, *DA = arg.shape
        arg = torch.tile(arg, [cotangent_vectors, *[1]*len(DA)])
        _args.append(arg)
        
    # sample cotangent vectors
    vectors = torch.randn_like(samples)
    if noise_type == 'rademacher':
        vectors = vectors.sign()
        
    f = lambda x: drift(t, x, *_args)
    _, vjp_func = vjp(f, samples)
    divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
    return divergence
