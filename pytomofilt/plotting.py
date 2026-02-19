import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyshtools

from . import sh_tools as sh

def plot_shcoefs(coefs, r=None, title='', quantity='', cmap='RdBu',
                      coast_color='k', projection=ccrs.Robinson(),
                      scale_factor=None, lmax=None, levels=None, vmin=None,
                      vmax=None, extend='neither', fig=None, ax=None):
    """
    Plot a set of spherical harmonic coefficients `coef`.  Label the radius with `r` (km)
    and the value with `quantity`.  Harmonics are truncated at degree `lmax`.
    If `fig` and `ax` are supplied, then the plot is added to the provided matplotlib
    figure and axis handles, respectively.
    """

    assert ( (ax is not None) and (fig is not None) or
             (ax is     None) and (fig is     None)), "Arg error: Axis set, but fig is None"

    if lmax is None:
        assert coefs.shape[1] == coefs.shape[2], "SH coefficients are not the correct shape"
        lmax = coefs.shape[1] - 1

    # For very "clean" coefficents (i.e. many zeros) the contour function
    # messes up and we get big chunks of the map with nothing plotted.
    # Avoid this by adding a small value to all coefficents.
    # See issue 42 on github.
    coefs = coefs + 1E-15

    # Convert coefs from RTS to SH format
    sh_coefs = sh.rts_to_sh(coefs)

    # Set plot area - fixed for now but we could use this to look at a local
    # area
    north = 90.0
    south = -90.0
    west = 0.0
    east = 360.0
    interval = 3.0
    # NB - arange goes from low to high, but we need N to S so reverse output.
    lats = np.arange(south, north+interval, interval)[::-1]
    lons = np.arange(west, east+interval, interval)
    grid = pyshtools.expand.MakeGrid2D(sh_coefs, interval, lmax, norm=4, north=north, south=south,
               east=east, west=west)
    if scale_factor is not None:
        grid = grid * scale_factor
    lons, lats = np.meshgrid(lons, lats)

    if ax is None and fig is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection),
                           figsize=(15,5))

    ax.set_global() # wrapping
    # No idea why, but we need "PlateCarree" here even if we use something
    # else (e.g. Robinson) for the projection above. This is odd but see
    # http://scitools.org.uk/cartopy/docs/v0.5/matplotlib/introductory_examples/
    # ... 03.contours.html
    h = ax.contourf(lons, lats, grid, 100,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, extend=extend)

    ax.coastlines(color=coast_color)
    if projection == ccrs.PlateCarree():
        ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
        ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    cb = fig.colorbar(h, ax=ax, orientation='vertical', fraction=0.02)
    cb.set_label(quantity, size=15)
    cb.ax.tick_params(axis='both', which='major', labelsize=15)

    if r is None:
        ax.set_title('{}'.format(title), size=15)
    else:
        ax.set_title('{} at {} km'.format(title, r), size=15)