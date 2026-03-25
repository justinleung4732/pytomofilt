#!/usr/bin/env python
"""
Functions to test the resolution of tomographic filters using synthetic models with spikes at depth.
"""
from pathlib import Path
from typing_extensions import Annotated

import cartopy.crs as ccrs
import pyshtools as shtools
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import typer

from .filter_models import tomographic_model_from_path
from . import model
from . import sh_tools as sh
from . import spline
from .plotting import plot_shcoefs


_rcmb = 3480.0
_rmoho = 6346.691

def _default_radii(rmin=_rcmb, rmax=_rmoho):
    # Magic numbers from S20RTS model defs (and everything else)
    KNOT_RADII = np.array([-1.00000, -0.78631, -0.59207, -0.41550, -0.25499, -0.10909,
                            0.02353, 0.14409, 0.25367, 0.35329, 0.44384, 0.52615,
                            0.60097, 0.66899, 0.73081, 0.78701, 0.83810, 0.88454,
                            0.92675, 0.96512, 1.00000])
    knots_r = (rmax - rmin) / 2.0 * KNOT_RADII + (rmin + rmax) / 2.0
    return knots_r


def create_lateral_delta_function(xlat, xlon, lmax):
    """
    Create a delta function at the given location (xlat, xlon) and expand it in spherical harmonics up to degree lmax.

    Parameters
    ----------
    xlat : float
        Latitude in degrees (-90 to 90).
    xlon : float
        Longitude in degrees (0 to 360 or -180 to 180).
    lmax : int
        Maximum spherical harmonic degree for expansion.

    Returns
    -------
    numpy.ndarray
        Real-valued spherical harmonic coefficients of the delta function,
        with shape (lmax+1, lmax+1) in orthonormal normalization.
    """
    theta = (90 - xlat)
    ylm = shtools.expand.spharm(lmax,theta,xlon, normalization='ortho', kind='real')
    return ylm


def create_radial_delta_function(r0, knots):
    """
    Compute spline coefficients for a Dirac delta approximation at x0.
    
    Parameters
    ----------
    r0 : float
        The location of the delta (must be within min(knots) to max(knots)).
    knots : array_like
        The knot points for the spline basis (same as in calculate_splines).
    
    Returns
    -------
    np.ndarray
        Coefficients c_i = B_i(r0) for each basis spline i.
    """
    # Ensure r0 is in range
    if not (min(knots) <= r0 <= max(knots)):
        raise ValueError("r0 must be within the knot range.")
    
    # Get the spline basis (as in calculate_splines)
    splines = spline.calculate_splines(knots)
    
    # Evaluate all basis functions at r0
    # splines(x0) returns shape (n_knots,) since y was identity
    coefs = splines(r0)  # c_i = B_i(r0)
    
    return coefs, splines


# get coefficients for delta function at spline depths
def resolution_test_bg_spike(
        r: Annotated[float, typer.Argument(help="Radius in km of the spike, between 3480 and 6346.691")],
        xlat: Annotated[float, typer.Argument(help="Latitude of the spike, between -90 and 90 degrees")],
        xlon: Annotated[float, typer.Argument(help="Longitude of the spike, between -180 and 360 degrees")],
        tomographic_model: Annotated[Path, typer.Argument(help="Path to a directory containing a tomographic model and filter")]
    ):
    """
    Resolution test for a background spike model. Creates a model with a spike at the given
    location and applies the tomographic filter to it.

    Parameters
    ----------
    r : float
        Radius in km of the spike, between 3480 and 6346.691
    xlat : float
        Latitude of the spike, between -90 and 90 degrees
    xlon : float
        Longitude of the spike, between -180 and 360 degrees.
    tomographic_model : Path
        Path to a directory containing a tomographic model and filter
    """
    assert _rcmb < r < _rmoho, "Radius must be between the core-mantle boundary and the Moho"
    assert -90 <= xlat <= 90, "Latitude must be between -90 and 90 degrees"
    assert -180 <= xlon <= 360, "Longitude must be between -180 and 360 degrees"

    # Build tomographic model reference
    tomographic_model_spec = tomographic_model_from_path(tomographic_model)
    ref_model = model.RTS_Model.from_file(tomographic_model_spec.coef_file)
    ref_model.filter_from_file(tomographic_model_spec.evec_file, tomographic_model_spec.weights_file,
                                0.2,verbose=True)
    lmax = ref_model.lmax

    # Create a delta function at the given location in spherical harmonics up to degree lmax.
    ylm = create_lateral_delta_function(xlat, xlon, lmax)
    ylm = sh.sh_to_rts(ylm)

    # Calculate the spline coefficients at the knot points
    knots_r = _default_radii(rmin=_rcmb, rmax=_rmoho)
    r_coefs, knot_splines = create_radial_delta_function(r, knots_r)
    spline_coefs = np.einsum('i,jkl->ijkl', r_coefs, ylm)  # shape (n_knots, 2, lmax+1, lmax+1)

    # Build a model object from the spline coefficients
    bg_spike_model = model.RTS_Model(lmax=lmax, rmin=_rcmb, rmax=_rmoho, knots=knots_r)
    bg_spike_model.coefs = spline_coefs

    print("Filtering!")
    filtered_bg_spike_model = ref_model.filter(bg_spike_model)

    # Plotting
    print("Plotting!")
    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    ax = []
    for i in range(3):
        ax.append(fig.add_subplot(gs[i, 0], projection=ccrs.Robinson()))
    
    # Rectilinear plot spanning the right column (all rows)
    ax_rect = fig.add_subplot(gs[:, 1])

    data1 = ylm
    data2 = spline.evaluate_coefs_at_r(r, knot_splines, spline_coefs)
    data3 = spline.evaluate_coefs_at_r(r, knot_splines, filtered_bg_spike_model.coefs)

    # Plot the spherical harmonic coefficients of the input delta function, the reparameterised
    # delta function, and the final filtered delta function
    _,_,h = plot_shcoefs(data1,
                 fig=fig, ax=ax[0],
                 cmap = 'Greys',
                 title="Input delta function in SH")
    plot_shcoefs(data2,
                 fig=fig, ax=ax[1],
                 cmap = 'Greys',
                 title="Reparameterised delta function",
                 levels = h.levels) # use same levels as data1 for comparison
    plot_shcoefs(data3,
                 fig=fig, ax=ax[2],
                 cmap = 'Greys',
                 title="Final filtered delta function",
                 levels = h.levels) # use same levels as data1 for comparison

    # Plot radial profile of the spike before and after filtering
    r_eval = np.linspace(_rcmb+1, _rmoho-1, 100)
    
    # Calculate value at spike lat,lon for each depth
    coefs_pre_filter = spline.evaluate_coefs_at_r(r_eval, knot_splines, spline_coefs)
    coefs_post_filter = spline.evaluate_coefs_at_r(r_eval, knot_splines, filtered_bg_spike_model.coefs)
    spike_pre_filter = np.zeros_like(r_eval)
    spike_post_filter = np.zeros_like(r_eval)
    for i,(pre,post) in enumerate(zip(coefs_pre_filter, coefs_post_filter)):
        pre = sh.rts_to_sh(pre)
        post = sh.rts_to_sh(post)
        spike_pre_filter[i] = shtools.expand.MakeGridPoint(pre,xlat,xlon,norm=4)
        spike_post_filter[i] = shtools.expand.MakeGridPoint(post,xlat,xlon,norm=4)
    
    # Plot the spike value as a function of depth before and after filtering
    ax_rect.plot(spike_pre_filter, r_eval, c='k', label="Pre-filter")
    ax_rect.plot(spike_post_filter, r_eval, c='r', label="Post-filter")
    ax_rect.set_title("Spike value at location as a function of depth")
    ax_rect.set_xlabel("Spike value")
    ax_rect.set_ylabel("Radius (km)")
    ax_rect.invert_yaxis()
    ax_rect.legend()

    plt.show()


if __name__ == "__main__":
    resolution_test_bg_spike(xlat=0, xlon=0, r=4371,
                             tomographic_model=Path("/Users/justinleung/code/pytomofilt/data/S12RTS"))