from torch import nn


class NullNet(nn.Module):

    def __init__(self, isenergy=False, *args, **kwargs):
        super().__init__()

        self.hyperparameters = {"nn_is_energy": isenergy}

    def forward(self, t, x, *args, **kwargs):
        raise RuntimeError("This is a null model and should not be called.")
