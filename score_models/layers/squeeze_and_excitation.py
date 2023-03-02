import torch
from torch import nn
from torch.nn import functional as F


class SqueezeAndExcite(nn.Module):
    """
    Implementation of the Sqeeze and Excite module, a form of channel attention
    originally described in Hu et al (2019). Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507
    """
    def __int__(self, channels, hidden_units):
        super(SqueezeAndExcite, self).__int__()
        self.excite_network = nn.Sequential(
            nn.Linear(channels, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        z = torch.mean(x, dim=(2, 3))  # Squeeze operation is a global average
        z = self.excite_network(z)  # compute channel importance
        s = F.sigmoid(z).view(B, C, 1, 1)
        return s * x  # scale channels
