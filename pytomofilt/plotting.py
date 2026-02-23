from typing import Optional, Tuple, Any

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import numpy as np
import pyshtools
from scipy.interpolate import griddata

from . import sh_tools as sh


def plot_shcoefs(coefs: np.ndarray,
                 r: Optional[float] = None,
                 title: str = '',
                 quantity: str = '',
                 cmap: str = 'RdBu',
                 coast_color: str = 'k',
                 projection: ccrs.CRS = ccrs.Robinson(),
                 scale_factor: Optional[float] = 1.0,
                 lmax: Optional[int] = None,
                 levels: Optional[np.ndarray] = None,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 extend: str = 'neither',
                 fig: Optional[plt.Figure] = None,
                 ax: Optional[plt.Axes] = None,
                 figsize: Tuple[int, int] = (15, 5),
                 labelsize: int = 15
                 ) -> Tuple[plt.Figure, plt.Axes, Any]:
    """
    Plot a set of spherical harmonic coefficients `coefs`. Label the radius with `r` (km)
    and the value with `quantity`.  Harmonics are truncated at degree `lmax`.
    If `fig` and `ax` are supplied, then the plot is added to the provided matplotlib
    figure and axis handles, respectively.
    
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
    cmap : str
        Colormap for plotting grid filled contours. Default is RdBu
    coast_color : str
        Color for coastlines. Default is black.
    projection : ccrs.CRS
        Geographic project used for figure. Default is ccrs.Robinson()
    scale_factor: float, optional
        Scale factor multiplied to the grid before plotting. Default is 1
    levels, vmin, vmax, extend :
        Same as those in plt.contourf
    fig : plt.Figure
        Figure object of the plot.
    ax : plt.Axes
        Axes for the plot.
    figsize : tuple
        Figure size. Default is (15,5) 
    labelsize : int
        Labelsize for axes and colorbar ticklabels. Default is 15.

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
    interval = 3
    north = 90
    south = -90
    west = 0
    east = 360
    
    grid = pyshtools.expand.MakeGrid2D(
        sh_coefs, interval, lmax, norm=4,
        north=north, south=south, east=east, west=west
    )

    # longitudes and latitudes (structured grid)
    lats = np.arange(south, north + interval, interval)[::-1]  # N->S ordering preserved from original
    lons = np.arange(west, east + interval, interval)

    fig, ax, mappable = _plot_contour_map(lons, lats, grid, r=r, title=title, quantity=quantity,
                                          cmap=cmap, coast_color=coast_color, 
                                          projection=projection,
                                          scale_factor=scale_factor, levels=levels, 
                                          vmin=vmin, vmax=vmax, extend=extend, fig=fig, ax=ax,
                                          figsize=figsize, labelsize=labelsize)

    return fig, ax, mappable


def plot_grid(lons: np.ndarray,
              lats: np.ndarray,
              vals: np.ndarray,
              r: Optional[float] = None,
              title: str = '',
              quantity: str = '',
              cmap: str = 'RdBu',
              coast_color: str = 'k',
              projection: ccrs.CRS = ccrs.Robinson(),
              scale_factor: Optional[float] = 1.0,
              levels: Optional[np.ndarray] = None,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None,
              extend: str = 'neither',
              fig: Optional[plt.Figure] = None,
              ax: Optional[plt.Axes] = None,
              figsize: Tuple[int, int] = (15, 5),
              labelsize: int = 15
              ) -> Tuple[plt.Figure, plt.Axes, Any]:
    """
    Plot a grid of values defined at (lons, lats). 
    If `fig` and `ax` are supplied, then the plot is added to the provided matplotlib
    figure and axis handles, respectively.

    Parameters
    ----------
    lons, lats : array-like
        1D arrays of point coordinates (degrees).
    values : array-like
        Values corresponding to the coordinates.
    r : float, optional
        Radius (km) to display in title.
    title : str
        Title for the plot.
    quantity : str
        Label for colorbar.
    cmap : str
        Colormap for plotting grid filled contours. Default is RdBu
    coast_color : str
        Color for coastlines. Default is black.
    projection : ccrs.CRS
        Geographic project used for figure. Default is ccrs.Robinson()
    scale_factor: float, optional
        Scale factor multiplied to the grid before plotting. Default is 1
    levels, vmin, vmax, extend :
        Same as those in plt.contourf
    fig : plt.Figure
        Figure object of the plot.
    ax : plt.Axes
        Axes for the plot.
    figsize : tuple
        Figure size. Default is (15,5) 
    labelsize : int
        Labelsize for axes and colorbar ticklabels. Default is 15.

    Returns
    -------
    fig, ax, mappable
    """

    assert len(lons) == len(lats) == len(vals), "Length of lons, lats and grids must be equal"

    eq_lats = np.arange(-90, 90 + 3, 3)[::-1]  # N->S ordering preserved from original
    eq_lons = np.arange(0, 360, 3)

    # If there are too many points, reduce the number of them used for interpolation
    limit = 50000
    n = int(np.ceil(len(lons)/limit))
    grid = griddata((lons[::n], lats[::n]), vals[::n], (eq_lons[None,:], eq_lats[:,None]), method='linear')
    grid, eq_lons = add_cyclic_point(grid, coord=eq_lons) # Avoid white line between lon=0 and 360

    fig, ax, mappable = _plot_contour_map(eq_lons, eq_lats, grid, r=r, title=title, 
                                          quantity=quantity, cmap=cmap, coast_color=coast_color, 
                                          projection=projection,
                                          scale_factor=scale_factor, levels=levels, 
                                          vmin=vmin, vmax=vmax, extend=extend, fig=fig, ax=ax,
                                          figsize=figsize, labelsize=labelsize)

    return fig, ax, mappable


def _plot_contour_map(lons: np.ndarray,
                      lats: np.ndarray,
                      grid: np.ndarray,
                      r: Optional[float] = None,
                      title: str = '',
                      quantity: str = '',
                      cmap: str = 'RdBu',
                      coast_color: str = 'k',
                      projection: ccrs.CRS = ccrs.Robinson(),
                      scale_factor: Optional[float] = 1.0,
                      levels: Optional[np.ndarray] = None,
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      extend: str = 'neither',
                      fig: Optional[plt.Figure] = None,
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[int, int] = (15, 5),
                      labelsize: int = 15
                      ) -> Tuple[plt.Figure, plt.Axes, Any]:
    """
    Private function to plot contour map from a gridded set of values, grid positions 
    are given by lons and lats. Called by plot_shcoefs and plot_grid.
    
    Parameters
    ----------
    lons, lats : array-like
        1D arrays of the list of longitudes and latitudes at which the grid is evaluated.
    grid : array-like
        A 2D array values corresponding to the grid set by lons and lats.
    Other variables: see documentation in plot_grid or plot_shcoefs

    Returns
    -------
    fig, ax, mappable
    """

    grid = grid * scale_factor

    if (ax is not None) ^ (fig is not None):
        raise ValueError("Both `fig` and `ax` must be provided together, or neither.")
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=figsize)

    # No idea why, but we need "PlateCarree" here even if we use something
    # else (e.g. Robinson) for the projection above. This is odd but see
    # http://scitools.org.uk/cartopy/docs/v0.5/matplotlib/introductory_examples/
    # ... 03.contours.html
    lon2d, lat2d = np.meshgrid(lons, lats)
    mappable = ax.contourf(lon2d, lat2d, grid, 100,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, extend=extend)

    # Set plot as global and add coastlines
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

    # title formatting
    if r is None:
        ax.set_title(f'{title}', size=15)
    else:
        ax.set_title(f'{title} at {np.round(r,3)} km', size=15)

    return fig, ax, mappable
