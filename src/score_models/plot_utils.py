from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scienceplots # pip install SciencePlots
import colorcet as cc # pip install colorcet
import pylab
import torch
from scipy.interpolate import interpn
from scipy.special import logsumexp
from matplotlib.colors import Normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use('dark_background')
plt.style.use('science')
params = {
         'figure.figsize': (4, 4),
         'axes.labelsize': 18,
         'axes.titlesize': 24,
         'ytick.labelsize' :20,
         'xtick.labelsize' :20,
         'legend.fontsize': 16,
         'xtick.major.size': 8,
         'xtick.minor.size': 4,
         'ytick.major.size': 8,
         'ytick.minor.size': 4,
         'text.usetex': True,
         'text.latex.preamble': r'\usepackage{bm}',
         }
pylab.rcParams.update(params)
cmap = cc.cm.fire

def plot_density(
        logp_fn: Callable,
        fig=None, 
        ax=None, 
        extent=(-2, 2, -2, 2),
        dx=0.025, 
        dy=0.025,
        colorbar=False, 
        cmap=cmap, 
        vmin=None, 
        vmax=None, 
        **kwargs):
    """
    Plot the density of a log probability function
    """
    # Generate a grid of points
    (xmin, xmax, ymin, ymax) = extent
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    n = x.size
    m = y.size
    points = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape((-1, 2))
    # Compute log probability and normalize to get the density
    logp = logp_fn(torch.tensor(points).to(DEVICE).float()).detach().numpy().reshape([m, n])
    p = np.exp(logp - logsumexp(logp + np.log(dx) + np.log(dy)))
    # Plot the density
    norm = Normalize(vmin=p.min() if vmin is None else vmin, vmax=p.max() if vmax is None else vmax)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(p, extent=extent, cmap=cmap, norm=norm, aspect="auto", origin="lower", **kwargs)
    if colorbar and fig is not None:
        cax = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
        cax.set_ylabel(r'$p(\mathbf{x})$')
    return ax

def plot_scatter(
        points, 
        fig=None, 
        ax=None, 
        bins=20, 
        sort=True, 
        norm=None, 
        ticks=None, 
        colorbar=False, 
        cmap=cmap, 
        **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    x = points[:, 0]
    y = points[:, 1]
    # Compute histogram and intepolate density
    data, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    x_bins = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_bins = 0.5 * (y_edges[1:] + y_edges[:-1])
    z = interpn((x_bins, y_bins), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    if ax is None:
        fig, ax = plt.subplots()
    if norm is None:
        norm = Normalize(vmin=z.min(), vmax=z.max())
    ax.scatter(x, y, c=z, cmap=cmap, norm=norm, **kwargs)
    if fig is not None and colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        box = ax.get_position()
        cax = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.02, box.height])
        fig.colorbar(sm, cax=cax, ticks=ticks)
        cax.set_ylabel(r'$p(\mathbf{x})$')
    return ax


def plot_score(
        score_fn: Callable,
        fig=None,
        ax=None,
        extent=(-2, 2, -2, 2),
        n=20, 
        scale=None, 
        width=0.007,
        colorbar=False,
        cmap=cmap,
        ):
    """
    Plot a vector field
    """
    # Generate a grid of points
    (xmin, xmax, ymin, ymax) = extent
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    points = np.stack(np.meshgrid(x, y), axis=-1).reshape((-1, 2))
    g = score_fn(torch.tensor(points).to(DEVICE).float()).detach().numpy().reshape([n, n, 2])
    colors = np.sqrt(g[..., 0]**2 + g[..., 1]**2).ravel()
    if ax is None:
        fig, ax = plt.subplots()
    norm = Normalize()
    ax.quiver(x, y, g[..., 0], g[..., 1], color=cmap(norm(colors)), scale=scale, width=width)
    if fig is not None and colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        box = ax.get_position()
        cax = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.02, box.height])
        fig.colorbar(sm, cax=cax)
        cax.set_ylabel(r"$\lVert \nabla \log p(\mathbf{x}) \rVert$")
    return ax

def plot_contours(
        logp_fn: Callable,
        fig=None,
        ax=None,
        extent=(-2, 2, -2, 2),
        ci: tuple = (0.68, 0.95, 0.99),
        dx=0.025, 
        dy=0.025,
        cmap=cc.cm.fire,
        ):
    """
    Plot density contours
    """
    (xmin, xmax, ymin, ymax) = extent
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    n = x.size
    m = y.size
    points = np.stack(np.meshgrid(x, y), axis=-1).reshape((-1, 2))
    # Compute log probability and normalize to get the density
    logp = logp_fn(torch.tensor(points).to(DEVICE)).detach().numpy().reshape([m, n])
    p = np.exp(logp - logsumexp(logp))
    # Compute the cumulative probability
    cumul = np.sort(p.ravel() * dx * dy)[::-1].cumsum()
    ps = []
    for _ci in ci:
        p_at_ci = np.sort(p.ravel())[::-1][np.argmin((cumul - _ci)**2)]
        ps.append(p_at_ci)
    colors = [cmap(i/(len(ci)-1)) for i in range(len(ci))]
    contours = ax.contour(x, y, p, levels=ps[::-1], colors=colors, linewidths=2, linestyles="--")
    if ax is None:
        fig, ax = plt.subplots()
    def fmt(x):
        ci = cumul[np.argmin((x - np.sort(p.ravel())[::-1])**2)]
        s = f"{ci*100:.1f}"
        if s.endswith("0"):
            s = f"{ci*100:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
    ax.clabel(contours, contours.levels, inline=True, fmt=fmt, fontsize=10)
    return ax

