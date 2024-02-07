import torch
from torch import nn
import numpy as np
"""
Implementation detail: torch.bmm (batched matmul) is preferred to torch.einsum 
since the latter has some reported speed issues.
"""


class SelfAttentionBlock(nn.Module):
    """
    Compute self attention over channels, with skip connection.
    """
    def __init__(self, channels, init_scale=1e-2, dimensions=2):
        """

        :param channels:
        :param init_scale: A number between 0 and 1 which scales the variance of initialisation
            of the output layer. 1 is the usual Glorot initialisation. The default is 0.01 so that
            the output of the attention head is squashed in favor of the skip connection at the beginning of training.
        """
        super(SelfAttentionBlock, self).__init__()
        if dimensions == 1:
            conv = nn.Conv1d
        elif dimensions == 2:
            conv = nn.Conv2d
        elif dimensions == 3:
            conv = nn.Conv3d
        assert (init_scale <= 1) and (init_scale > 0)
        self.to_qkv = conv(in_channels=channels, out_channels=3*channels, kernel_size=1)
        self.to_out = conv(in_channels=channels, out_channels=channels, kernel_size=1)

        # initialisation
        with torch.no_grad():
            bound = init_scale / channels ** (1 / 2)
            self.to_out.weight.uniform_(-bound, bound)
            self.to_out.bias.zero_()
            bound = 1 / channels ** (1 / 2)
            self.to_qkv.weight.uniform_(-bound, bound)
            self.to_qkv.bias.zero_()

    def __call__(self, x):
        B, C, *D = x.shape
        q, k, v = torch.tensor_split(self.to_qkv(x), 3, dim=1)

        q = q.permute(0, *range(2, len(D)+2), 1).view(B, np.prod(D), C)
        k = k.view(B, C, np.prod(D))
        v = v.permute(0, *range(2, len(D)+2), 1).view(B, np.prod(D), C)

        w = torch.bmm(q, k) * (C**(-0.5))  # scaled channel attention matrix, QK^T / sqrt(d)
        w = torch.softmax(w, dim=-1)
        attention = torch.bmm(w, v)
        attention = attention.view(B, *D, C).permute(0, -1, *range(1, len(D)+1))
        return self.to_out(attention) + x


class ScaledAttentionLayer(nn.Module):
    """
    Simple self attention mechanism, with MLP and no skip connections for MLP network
    """
    def __init__(self, dimensions):
        super().__init__()
        self.query = nn.Linear(in_features=dimensions, out_features=dimensions)
        self.key = nn.Linear(in_features=dimensions, out_features=dimensions)
        self.value = nn.Linear(in_features=dimensions, out_features=dimensions)
        self.to_out = nn.Linear(in_features=dimensions, out_features=dimensions)

        # Initialization
        with torch.no_grad():
            bound = 1 / dimensions ** (1 / 2)
            for layer in (self.query, self.key, self.value):
                layer.weight.uniform_(-bound, bound)
                layer.bias.zero_()
            bound = 1 / dimensions ** (1 / 2)
            self.to_out.weight.uniform_(-bound, bound)
            self.to_out.bias.zero_()

    def __call__(self, query, context):
        B, C_out, D = query.shape
        B, C_in, D = context.shape
       
        query = query.view(B * C_out, D)
        query = self.query(query).view(B, C_out, D)
        value = self.value(context.view(B * C_in, D)).view(B, C_in, D)
        scores = torch.bmm(query, context.transpose(1, 2)) / D**(0.5)  # scaled attention matrix, QK^T / sqrt(d)
        scores = torch.softmax(scores, dim=-1)
        attention = torch.bmm(scores, value)
        h = self.to_out(attention)
        return h
