from score_models.ncsnpp import NCSNpp
import torch
import numpy as np
from tqdm import tqdm


class NCSNppLog(NCSNpp):
    def __init__(
            self,
            beta0,
            beta1,
            channels=1,
            sigma_min=1e-1,
            sigma_max=50,
            nf=128,
            ch_mult=(1, 1, 2, 2, 2, 2, 2),
            num_res_blocks=2,
            activation_type="swish",
            dropout=0.,
            resample_with_conv=True,
            fir=True,
            fir_kernel=(1, 3, 3, 1),
            skip_rescale=True,
            progressive="output_skip",
            progressive_input="input_skip",
            init_scale=1e-2,
            fourier_scale=16.,
            resblock_type="biggan",
            combine_method="sum",
            attention=True,
            **kwargs
          ):
        super().__init__(
            channels=channels,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            nf=nf,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            activation_type=activation_type,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
            skip_rescale=skip_rescale,
            progressive=progressive,
            progressive_input=progressive_input,
            init_scale=init_scale,
            fourier_scale=fourier_scale,
            resblock_type=resblock_type,
            combine_method=combine_method,
            attention=attention,
            **kwargs)
        self.beta0 = beta0
        self.beta1 = beta1

    def score(self, x, t, detrended=True):
        """
        Assumes x is coming from the trended log VESDE
        x = torch.log(x + self.beta0 + self.beta1 * t.view(B, *[1]*len(D)))

        detrended is a keyword to make us if x comes directly from the SDE or
        has been detrended.
        **NOTE**
        During training, x must be detrended outside of this function, otherwise the network does
        not converge to the solution. This is related to the Jacobian trace computation, which requires
        computing a gradient of this function wrt its input.
        """
        B, *D = x.shape
        broadcast = [-1, *[1]*len(D)]
        if not detrended:
            x = x - torch.log(self.beta0 + self.beta1*t.view(*broadcast))
        # We need to retrend x to compute the scale of the score.
        w = torch.exp(x + torch.log(self.beta0 + self.beta1*t.view(*broadcast)))
        return self.forward(x, t) / self.sde.sigma(t).view(*broadcast) * w

    # @torch.no_grad() # TODO chang this method
    # def sample(self, size, N: int = 1000, device=DEVICE):
    #     assert len(size) == 4
    #     assert size[1] == self.channels
    #     assert N > 0
    #     # A simple Euler-Maruyama integration of VESDE
    #     x = torch.randn(size).to(device)
    #     dt = -1.0 / N
    #     t = torch.ones(size[0]).to(DEVICE)
    #     broadcast = [-1, 1, 1, 1]
    #     for _ in tqdm(range(N)):
    #         t += dt
    #         drift, diffusion = self.sde.sde(x, t)
    #         score = self.score(x, t)
    #         drift = drift - diffusion.view(*broadcast)**2 * score
    #         z = torch.randn_like(x)
    #         x_mean = x + drift * dt
    #         x = x_mean + diffusion.view(*broadcast) * (-dt)**(1/2) * z
    #     return x_mean
