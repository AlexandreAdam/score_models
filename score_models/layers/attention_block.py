import torch
from torch import nn
"""
Implementation detail: torch.bmm (batched matmul) is preferred to torch.einsum 
since the latter has some reported speed issues.
"""


class SelfAttentionBlock(nn.Module):
    """
    Compute self attention over channels, with skip connection.
    """
    def __init__(self, channels, init_scale=1e-2):
        """

        :param channels:
        :param init_scale: A number between 0 and 1 which scales the variance of initialisation
            of the output layer. 1 is the usual Glorot initialisation. The default is 0.01 so that
            the output of the attention head is squashed in favor of the skip connection at the beginning of training.
        """
        super(SelfAttentionBlock, self).__init__()
        assert (init_scale <= 1) and (init_scale > 0)
        self.to_qkv = nn.Conv2d(in_channels=channels, out_channels=3*channels, kernel_size=1)
        self.to_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        # initialisation
        with torch.no_grad():
            bound = init_scale / channels ** (1 / 2)
            self.to_out.weight.uniform_(-bound, bound)
            self.to_out.bias.zero_()
            bound = 1 / channels ** (1 / 2)
            self.to_qkv.weight.uniform_(-bound, bound)
            self.to_qkv.bias.zero_()

    def __call__(self, x):
        B, C, H, W = x.shape
        q, k, v = torch.tensor_split(self.to_qkv(x), 3, dim=1)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)

        w = torch.bmm(q, k) * (C**(-0.5))  # scaled channel attention matrix, QK^T / sqrt(d)
        w = torch.softmax(w, dim=-1)
        attention = torch.bmm(w, v)
        attention = attention.view(B, H, W, C).permute(0, 3, 1, 2)
        return self.to_out(attention) + x


# This completely equivalent formulation might be very useful
# when we consider conditioning this layer on the noise level.
class AlternativeSelfAttentionBlock(nn.Module):
    def __init__(self, channels, scale_init=1e-2):
        super(AlternativeSelfAttentionBlock, self).__init__()
        self.query = nn.Linear(in_features=channels, out_features=channels)
        self.key = nn.Linear(in_features=channels, out_features=channels)
        self.value = nn.Linear(in_features=channels, out_features=channels)
        self.to_out = nn.Linear(in_features=channels, out_features=channels)

        # Initialization
        with torch.no_grad():
            bound = 1 / channels ** (1 / 2)
            for layer in (self.query, self.key, self.value):
                layer.weight.uniform_(-bound, bound)
                layer.bias.zero_()
            bound = scale_init / channels ** (1 / 2)
            self.to_out.weight.uniform_(-bound, bound)
            self.to_out.bias.zero_()

    def __call__(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        q = self.query(x).view(B, H * W, C)
        k = self.key(x).view(B, H * W, C).permute(0, 2, 1)
        v = self.value(x).view(B, H * W, C)

        w = torch.bmm(q, k) / C**(0.5)  # scaled attention matrix, QK^T / sqrt(d)
        w = torch.softmax(w, dim=-1)
        attention = torch.bmm(w, v)
        h = self.to_out(attention).view(B, H, W, C).permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)
        return x + h


if __name__ == '__main__':
    x = torch.randn([10, 4, 8, 8])
    print(x[0, 0, 0, 0], x[0, 0, 0, 1])
    att = SelfAttentionBlock(4)
    y = att(x)
    print(y[0, 0, 0, 0], y[0, 0, 0, 1])
    att = AlternativeSelfAttentionBlock(4)
    y = att(x)
    print(y[0, 0, 0, 0], y[0, 0, 0, 1])
