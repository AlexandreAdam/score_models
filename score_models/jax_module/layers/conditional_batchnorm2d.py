import torch.nn as nn
import torch


class ConditionalBatchNorm2d(nn.Module):
    """
    From Song & Ermon 2020. Original implementation
    (https://github.com/ermongroup/ncsn/blob/master/models/refinenet_dilated_baseline.py)
    """
    def __init__(self, num_features, num_classes=None, bias=True):
        """
        :param num_features: Number of channels of the image tensor
        :param num_classes: Number of denoising steps in the score autoencoder. If None, then we suppose the
            condition is sigma itself.
        """
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if num_classes is None:
            self.prepare_dim = lambda condition: condition.view(-1, 1)
            if self.bias:
                self.embed = nn.Linear(1, num_features * 2, bias=False)
            else:
                self.embed = nn.Linear(1, num_features, bias=False)
        else:  # the condition is the index of a schedule, we use an embedding layer in this case
            self.prepare_dim = lambda condition: condition
            if self.bias:
                self.embed = nn.Embedding(num_classes, num_features * 2)
            else:
                self.embed = nn.Embedding(num_classes, num_features)
        with torch.no_grad():
            if self.bias:
                self.embed.weight.data[:, :num_features].normal_(1, 0.02)   # Initialise scale at N(1, 0.02)
                self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
            else:
                self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, condition):
        """
        Condition should have shape [B] when index is discrete, and [B, 1] when it is continuous for this code to work.
        """
        condition = self.prepare_dim(condition)
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(condition).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(condition)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


if __name__ == '__main__':
    # Continuous case
    some_network = ConditionalBatchNorm2d(10, None, bias=False)
    some_input_image = torch.randn((10, 10, 32, 32))  # [B, C, H, W]
    time_variable = torch.randn((10,))
    some_network.forward(some_input_image, time_variable)

    # Discrete case
    some_network = ConditionalBatchNorm2d(10, 3)
    some_input_image = torch.randn((10, 10, 32, 32))  # [B, C, H, W]
    time_index = torch.randint(size=(10,), high=3)
    some_network.forward(some_input_image, time_index)
