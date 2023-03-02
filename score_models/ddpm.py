"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/
with slight modifications to make it work on continuous time.
"""
from torch import nn
import torch
import torch.nn.functional as F
from score_models.sde import VESDE
from score_models.definitions import DEVICE
from .utils import get_activation
from .layers import DDPMResnetBlock, SelfAttentionBlock, Conv2dSame, GaussianFourierProjection
import functools
from tqdm import tqdm


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1):
    """3x3 convolution with DDPM initialization."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=bias)
    return conv


class Downsample(nn.Module):
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
        self.Conv_0 = conv3x3(channels, channels, stride=2)
    self.with_conv = with_conv

  def forward(self, x):
    if self.with_conv:
        x = self.Conv_0(x)
    else:
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
    return x


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
        if self.with_conv:
            h = self.Conv_0(h)
        return h


class DDPM(nn.Module):
    def __init__(
            self,
            channels=1,
            sigma_min=1e-1,
            sigma_max=50,
            nf=128,
            activation_type="relu",
            ch_mult=(1, 1, 2, 2, 4, 4),
            num_res_blocks=2,
            resample_with_conv=True,
            dropout=0.,
            attention=True,
            **kwargs
    ):
        super().__init__()
        self.hyperparameters = {
            "channels": channels,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "nf": nf,
            "activation_type": activation_type,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "resample_with_conv": resample_with_conv,
            "dropout": dropout,
            "attention": attention
        }
        self.act = act = get_activation(activation_type=activation_type)
        self.attention = attention
        self.channels = channels
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.sde = VESDE(sigma_min, sigma_max)

        AttnBlock = SelfAttentionBlock
        ResnetBlock = functools.partial(DDPMResnetBlock, act=act, temb_dim=4 * nf, dropout=dropout)

        # Condition on continuous time
        modules = [GaussianFourierProjection(embed_dim=nf), nn.Linear(nf, nf * 4), nn.Linear(nf * 4, nf * 4)]
        with torch.no_grad():
            modules[1].bias.zero_()
            modules[2].bias.zero_()

        # Downsampling block
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resample_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.attention:
            modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resample_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t):
        modules = self.all_modules
        m_idx = 0
        temb = t
        for _ in range(3):
            temb = modules[m_idx](temb)
            m_idx += 1

        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        if self.attention:
            h = modules[m_idx](h)
            m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        return h

    def score(self, x, t):
        B, *D = x.shape
        return self.forward(x, t) / self.sde.sigma(t).view(B, *[1]*len(D))

    @torch.no_grad()
    def sample(self, size, N: int = 1000, device=DEVICE):
        assert len(size) == 4
        assert size[1] == self.channels
        assert N > 0
        # A simple Euler-Maruyama integration of VESDE
        x = torch.randn(size).to(device)
        dt = -1.0 / N
        t = torch.ones(size[0]).to(DEVICE)
        broadcast = [-1, 1, 1, 1]
        for _ in tqdm(range(N)):
            t += dt
            drift, diffusion = self.sde.sde(x, t)
            score = self.score(x, t)
            drift = drift - diffusion.view(*broadcast)**2 * score
            z = torch.randn_like(x)
            x_mean = x + drift * dt
            x = x_mean + diffusion.view(*broadcast) * (-dt)**(1/2) * z
        return x_mean