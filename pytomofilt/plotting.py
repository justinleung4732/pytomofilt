from typing import Optional, Tuple, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyshtools

from . import sh_tools as sh

def _make_fig_ax(projection: ccrs.CRS,
                 fig: Optional[plt.Figure],
                 ax: Optional[plt.Axes],
                 figsize: Tuple[int, int] = (15, 5)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create (fig, ax) if not provided. Ensures projection is set on the axis.

    Parameters
    ----------
    projection : ccrs.CRS
        Geographic project used for figure.
    fig : plt.Figure, optional
        Figure object of the plot.
    ax : plt.Axes, optional
        Axes for the plot.
    figsize : Tuple[int, int]
        Prescribed figure size. Default is (15,5).

    Returns
    -------
    fig, ax
    """
    if (ax is not None) ^ (fig is not None):
        raise ValueError("Both `fig` and `ax` must be provided together, or neither.")
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=figsize)
    return fig, ax


def _finalize_map(fig: plt.Figure, ax: plt.Axes, mappable,
                  quantity: str = '', coast_color: str = 'k',
                  projection: ccrs.CRS = ccrs.Robinson(),
                  labelsize: int = 15) -> None:
    """
    Finalising the map to plot common features: coastlines, ticks (if PlateCarree), colorbar, title font sizing.
    
    Parameters
    ----------
    fig : plt.Figure, optional
        Figure object of the plot.
    ax : plt.Axes, optional
        Axes for the plot.
    mappable : plt.mappable object
        
    quantity : str
        Label for colorbar.
    coast_color : str
        Color for coastlines. Default is black.
    projection : ccrs.CRS
        Geographic project used for figure. Default is ccrs.Robinson()
    labelsize : int
        Labelsize for axes and colorbar ticklabels. Default is 15.
    """
    ax.set_global()
    ax.coastlines(color=coast_color)

    # use PlateCarree ticks when plotting in that CRS
    if isinstance(projection, ccrs.PlateCarree):
        ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
        ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)

    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # colorbar: vertical on the right
    cb = fig.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.02)
    cb.set_label(quantity, size=labelsize)
    cb.ax.tick_params(axis='both', which='major', labelsize=labelsize)


def plot_shcoefs(coefs: np.ndarray,
                 r: Optional[float] = None,
                 title: str = '',
                 quantity: str = '',
                 cmap: str = 'RdBu',
                 coast_color: str = 'k',
                 projection: ccrs.CRS = ccrs.Robinson(),
                 scale_factor: Optional[float] = None,
                 lmax: Optional[int] = None,
                 levels: Optional[np.ndarray] = None,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 extend: str = 'neither',
                 fig: Optional[plt.Figure] = None,
                 ax: Optional[plt.Axes] = None,
                 interval: float = 3.0,
                 north: float = 90.0,
                 south: float = -90.0,
                 west: float = 0.0,
                 east: float = 360.0
                 ) -> Tuple[plt.Figure, plt.Axes, Any]:
    """
    Plot spherical-harmonic coefficients.

    Parameters
    ----------
    coefs : np.ndarray
        RTS-format coefficients (as in your code). Shape expected to match sh_tools.rts_to_sh.
    r : float, optional
        Radius (km) to display in title.
    title : str
        Title for the plot.
    quantity : str
        Label for colorbar.
    cmap, coast_color, projection, scale_factor, lmax, levels, vmin, vmax, extend:
        same semantics as your original function.
    interval : float
        Grid spacing in degrees used for MakeGrid2D.
    north, south, west, east : float
        Domain extents passed to MakeGrid2D.

    Returns
    -------
    fig, ax, mappable
    """

    # Determine lmax
    if lmax is None:
        if coefs.shape[1] != coefs.shape[2]:
            raise ValueError("SH coefficients appear to have invalid shape; please provide lmax.")
        lmax = coefs.shape[1] - 1

    # small regularization to avoid contour artifacts (preserves original behavior)
    coefs = coefs + 1e-15

    # Convert to pyshtools-compatible SH array
    sh_coefs = sh.rts_to_sh(coefs)

    # Build grid (MakeGrid2D expects the SH array, spacing and extents)
    grid = pyshtools.expand.MakeGrid2D(
        sh_coefs, interval, lmax, norm=4,
        north=north, south=south, east=east, west=west
    )

    if scale_factor is not None:
        grid = grid * scale_factor

    # longitudes and latitudes (structured grid)
    lats = np.arange(south, north + interval, interval)[::-1]  # N->S ordering preserved from original
    lons = np.arange(west, east + interval, interval)
    lon2d, lat2d = np.meshgrid(lons, lats)

    fig, ax = _make_fig_ax(projection, fig, ax)

    # No idea why, but we need "PlateCarree" here even if we use something
    # else (e.g. Robinson) for the projection above. This is odd but see
    # http://scitools.org.uk/cartopy/docs/v0.5/matplotlib/introductory_examples/
    # ... 03.contours.html
    mappable = ax.contourf(lon2d, lat2d, grid, 100,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, extend=extend)

    _finalize_map(fig, ax, mappable, quantity=quantity, coast_color=coast_color, projection=projection)

    # title formatting
    if r is None:
        ax.set_title(f'{title}', size=15)
    else:
        ax.set_title(f'{title} at {np.round(r,3)} km', size=15)

    return fig, ax, mappable


def plot_grid(lons: np.ndarray,
              lats: np.ndarray,
              grid: np.ndarray,
              r: Optional[float] = None,
              title: str = '',
              quantity: str = '',
              cmap: str = 'RdBu',
              coast_color: str = 'k',
              projection: ccrs.CRS = ccrs.Robinson(),
              scale_factor: Optional[float] = None,
              levels: Optional[np.ndarray] = None,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None,
              extend: str = 'neither',
              fig: Optional[plt.Figure] = None,
              ax: Optional[plt.Axes] = None
              ) -> Tuple[plt.Figure, plt.Axes, Any]:
    """
    Plot a grid of values defined at (lons, lats). 

    Parameters
    ----------
    lons, lats : array-like
        1D arrays of point coordinates (degrees).
    grid : array-like
        Values corresponding to the coordinates.

    Returns
    -------
    fig, ax, mappable
    """

    assert len(lons) == len(lats) == len(grid), "Length of lons, lats and grids must be equal"
    if scale_factor is not None:
        grid = grid * scale_factor

    fig, ax = _make_fig_ax(projection, fig, ax)

    # expect lons/lats and grid as 1D arrays of same length
    mappable = ax.tricontourf(lons, lats, grid, 100,
                                transform=ccrs.PlateCarree(),
                                cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, extend=extend)

    _finalize_map(fig, ax, mappable, quantity=quantity, coast_color=coast_color, projection=projection)

    if r is None:
        ax.set_title(f'{title}', size=15)
    else:
        ax.set_title(f'{title} at {np.round(r,3)} km', size=15)

    return fig, ax, mappable