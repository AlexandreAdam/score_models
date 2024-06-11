import torch.nn as nn
import torch


class ConditionalInstanceNorm2dPlus(nn.Module):
    """
    From Song & Ermon 2020. Original implementation
    (https://github.com/ermongroup/ncsn/blob/master/models/refinenet_dilated_baseline.py).

    Goal of this instance norm is to correct for color shift in output maps due to information
    about the spatial mean being loss across avery feature map for InstanceNorm.
    """
    def __init__(self, num_features, num_classes=None, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if num_classes is None:
            self.prepare_dim = lambda condition: condition.reshape(-1, 1)
            if self.bias:
                self.embed = nn.Linear(1, num_features * 3, bias=False)
            else:
                self.embed = nn.Linear(1, num_features * 2, bias=False)
        else:  # the condition is the index of a schedule, we use an embedding layer in this case
            self.prepare_dim = lambda condition: condition
            if self.bias:
                self.embed = nn.Embedding(num_classes, num_features * 3)
            else:
                self.embed = nn.Embedding(num_classes, num_features * 2)
        with torch.no_grad():
            if self.bias:
                self.embed.weight.data[:, :num_features].normal_(1, 0.02)   # Initialise scale at N(1, 0.02)
                self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
            else:
                self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, condition):
        condition = self.prepare_dim(condition)
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        m = torch.mean(means, dim=1, keepdim=True)
        v = torch.var(means, dim=1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(condition).chunk(3, dim=-1)
            h = h + means * alpha[..., None, None]
            out = gamma.reshape(-1, self.num_features, 1, 1) * h + beta.reshape(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(condition).chunk(2, dim=-1)
            h = h + means * alpha[..., None, None]
            out = gamma.reshape(-1, self.num_features, 1, 1) * h
        return out


if __name__ == '__main__':
    # Continuous case
    some_network = ConditionalInstanceNorm2dPlus(10, None, bias=False)
    some_input_image = torch.randn((10, 10, 32, 32))  # [B, C, H, W]
    time_variable = torch.randn((10,))
    some_network.forward(some_input_image, time_variable)

    # Discrete case
    some_network = ConditionalInstanceNorm2dPlus(10, 3)
    some_input_image = torch.randn((10, 10, 32, 32))  # [B, C, H, W]
    time_index = torch.randint(size=(10,), high=3)
    some_network.forward(some_input_image, time_index)
