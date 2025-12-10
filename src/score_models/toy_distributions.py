import torch
import numpy as np
from torch.distributions import constraints
from torch import distributions as tfd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def swiss_roll(modes=128, size=0.1, width=0.1, spread=0.7, device=DEVICE) -> tfd.Distribution:
    """
    Returns a swiss roll distribution from of a mixture of <modes> gaussian distributions
        :param modes: Number of modes in the mixture
        :param size: Size of the swiss roll
        :param width: Width of the modes in the swiss roll
        :param spread: Angular spread of the modes in the swiss roll
    """
    assert (spread > 0)
    t = 1.5 * np.pi * (1 + 2 * torch.linspace(0, 1, modes)**(spread))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    coords = size * torch.stack([x, y], dim=1).to(device)
    mixture = tfd.Categorical(probs=torch.ones(modes).to(device), validate_args=False)
    component = tfd.Independent(tfd.Normal(loc=coords, scale=width, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def five_swiss_rolls(modes=128, size=0.1, width=0.1, spread=0.7, offset=3, device=DEVICE):
    """
    Returns 5 swiss roll distribution arranged in the classic 5 dots pattern
        :param modes: Number of modes in the mixture
        :param size: Size of the swiss roll
        :param width: Width of the modes in the swiss roll
        :param spread: Angular spread of the modes in the swiss roll
        :param offset: Distance between the swiss rolls
    """
    assert (spread > 0)
    offsets = [[-offset, offset], [offset, -offset], [offset, offset], [-offset, -offset]]
    coords = []
    for k in range(5):
        t = 1.5 * np.pi * (1 + 2 * torch.linspace(0, 1, modes)**(spread))
        x = size * t * torch.cos(t)
        y = size * t * torch.sin(t)
        if k > 0:
            x += offsets[k-1][0]
            y += offsets[k-1][1]
        coords.append(torch.stack([x, y], dim=1))
    coords = torch.concat(coords, dim=0).to(device)
    mixture = tfd.Categorical(probs=torch.ones(5*modes).to(device), validate_args=False)
    component = tfd.Independent(tfd.Normal(loc=coords, scale=width, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def two_moons(modes=128, width=0.1, size=1, device=DEVICE) -> tfd.Distribution:
    """
    Returns a 2 moons distributions from a mixture of <modes> gaussian distributions
    :param modes: Number of modes inside each moon
    :param width: Width of the moons
    :param size: scales the coordinates by this amount
    """
    outer_circ_x = torch.cos(torch.linspace(0, np.pi, modes)) - .5
    outer_circ_y = torch.sin(torch.linspace(0, np.pi, modes)) - .25
    inner_circ_x = - torch.cos(torch.linspace(0, np.pi, modes)) + .5
    inner_circ_y = - torch.sin(torch.linspace(0, np.pi, modes)) + .25
    x = torch.concat([outer_circ_x, inner_circ_x])
    y = torch.concat([outer_circ_y, inner_circ_y])
    coords = size * torch.stack([x, y], dim=1).to(device)
    mixture = tfd.Categorical(probs=torch.ones(2*modes).to(device), validate_args=False)  # Uniform
    component = tfd.Independent(tfd.Normal(loc=coords, scale=width, validate_args=False), 1)  # Diagonal Multivariate Normal
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def checkerboard(squares: int = 4, size: float = None) -> tfd.Distribution:
    """
    Checkerboard distribution
        :param squares: Number of squares on a side
        :param size: Size of the checkerboard. By default, size = squares.
    """
    if size is None:
        size = squares
    square_size = float(size / squares)
    x = torch.arange(-squares // 2 + squares % 2, squares // 2, 1)
    xx, yy = torch.meshgrid(x, x)
    mask = (((xx % 2) & (yy % 2)) + (((xx + 1) % 2) & ((yy + 1) % 2))).bool()
    coords = torch.stack([xx[mask].ravel(), yy[mask].ravel()], dim=1).float().to(DEVICE)
    coords *= square_size
    mixture = tfd.Categorical(probs=torch.ones(squares**2 // 2), validate_args=False)
    component = tfd.Independent(tfd.Uniform(low=coords, high=coords + square_size, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def eggbox(
        modes: int = 16, 
        mode_width: float = 0.5, 
        box_size: float = None, 
        weights: tuple[float] = None, 
        device=DEVICE) -> tfd.Distribution:
    """
    Returns gaussian mixture equally spaces on the 2d plane.
        :param modes: Number of modes in the mixture
        :param box_size: Size egg box
        :param mode_width: Width of the modes
    """
    assert int(np.sqrt(modes))**2 == modes, f"modes = {modes} is not a square number"
    if box_size is None:
        box_size = modes**(1/2)
    x = torch.linspace(-1, 1, int(np.sqrt(modes))).to(device)
    x, y = torch.meshgrid(x, x, indexing="xy")
    coords = box_size * torch.stack([x.ravel(), y.ravel()], dim=1)
    if weights is None:
        weights = torch.ones(modes).to(device)
        weights /= weights.sum()
    else:
        weights = torch.tensor(weights).to(device)
        weights /= weights.sum()
    mixture = tfd.Categorical(probs=weights, validate_args=False)
    component = tfd.Independent(tfd.Normal(loc=coords, scale=mode_width, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def two_modes1D(mode_width=0.3, weights=None, mode_distance=1):
    """
    Returns a distribution with two modes
    """
    if weights is None:
        weights = torch.tensor([0.5, 0.5]).to(DEVICE)
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights).to(DEVICE)
    coords = torch.tensor([[-.5, .5]]).T.to(DEVICE) * mode_distance
    mixture = tfd.Categorical(probs=weights, validate_args=False)
    component = tfd.Independent(tfd.Normal(loc=coords, scale=mode_width, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


def two_modes2D(mode_width=0.3, weights=None):
    """
    Returns a distribution with two modes
    """
    coords = torch.tensor([[-0.5, 0.], [0.5, -0.]]).to(DEVICE)
    if weights is None:
        weights = torch.tensor([0.5, 0.5]).to(DEVICE)
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights).to(DEVICE)
    mixture = tfd.Categorical(probs=weights, validate_args=False)
    component = tfd.Independent(tfd.Normal(loc=coords.to(DEVICE), scale=mode_width, validate_args=False), 1)
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)
