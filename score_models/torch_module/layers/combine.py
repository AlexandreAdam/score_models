import torch
from .conv_layers import conv1x1


class Combine(torch.nn.Module):
    """Combine information from skip connections."""

    def __init__(self, in_ch, out_ch, method='cat', dimensions:int = 2):
        super().__init__()
        self.Conv_0 = conv1x1(in_ch, out_ch, dimensions=dimensions)
        assert method in ["cat", "sum"], f'Method {method} not recognized.'
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
