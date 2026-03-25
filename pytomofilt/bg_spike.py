#!/usr/bin/env python
"""
Functions to test the resolution of tomographic filters using synthetic models with spikes at depth.
"""
from pathlib import Path
from typing_extensions import Annotated

import cartopy.crs as ccrs
import pyshtools as shtools
import matplotlib.pyplot as plt
import numpy as np
import typer

from . filter_models import tomographic_model_from_path
from . import model
from . import sh_tools as sh
from . import spline
from . plotting import plot_shcoefs


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
    print(f"Setting up filter model for {tomographic_model_spec.name}.")
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

    fig, ax = plt.subplots(1,3, figsize=(20,8), subplot_kw=dict(projection=ccrs.Robinson()))
    # Calculate common colorscale limits
    data1 = ylm
    data2 = np.einsum('i,ijkl->jkl', knot_splines(r), spline_coefs)
    data3 = np.einsum('i,ijkl->jkl', knot_splines(r), filtered_bg_spike_model.coefs)
    
    #FIXME: colorbars not consistent across the three plots, need to calculate common limits
    plot_shcoefs(data1,
                 fig=fig, ax=ax[0],
                 title="Input delta function in spherical harmonics")
    plot_shcoefs(data2,
                 fig=fig, ax=ax[1],
                 title="Reparameterised delta function")
    plot_shcoefs(data3,
                 fig=fig, ax=ax[2],
                 title="Final filtered delta function")