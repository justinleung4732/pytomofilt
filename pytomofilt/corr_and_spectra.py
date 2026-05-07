#!/usr/bin/env python
"""
Functions to compute correlation and spectra of tomographic models.

These are used to provide a command line interface to computing the correlation and spectra of
tomographic models.
"""
from pathlib import Path
from typing_extensions import Annotated

import numpy as np
import pyshtools as shtools
import typer

from . model import RTS_Model
from . import sh_tools as sh
from . import plotting

_rcmb = 3480.0
_rmoho = 6346.691


def calc_spectra(
        filename: Annotated[Path, typer.Argument(help="Path to the .sph file containing the model data.")],
        rmin: Annotated[float, typer.Argument(help="Minimum radius for the model.")] = _rcmb, 
        rmax: Annotated[float, typer.Argument(help="Maximum radius for the model.")] = _rmoho,
        knots: Annotated[np.ndarray, typer.Argument(help="Radial knots for the model parameterization.")] = None,
        output_filename: Annotated[str, typer.Argument(help="Path to save the computed spectra.")] = ''
        ):
    """
    Compute the power spectrum of a parameterised geodynamic or tomographic model in a .sph file. 
    The power spectra is plotted, optionally saves results to a file.

    Parameters
    ----------
    filename : Path
        Path to the .sph file containing the model data.
    rmin : float, optional
        Minimum radius for the model, by default 3480 (Core-Mantle Boundary radius).
    rmax : float, optional
        Maximum radius for the model, by default 6346.691 (Mohorovičić discontinuity radius).
    knots : array-like, optional
        Radial knots for the model parameterization. If None, uses default knots from model files.
    output_filename : str, optional
        Path to save the computed spectra as a text file, by default path is empty.
    """
    model = RTS_Model.from_file(filename, rmin=rmin, rmax=rmax, knots=knots)
    spectra = np.empty_like(model.knots_r)
    for i, coef in enumerate(model.coefs):
        sh_coef = sh.rts_to_sh(coef)
        spectra[i] = shtools.spectralanalysis.spectrum(sh_coef, normalization='ortho')

    # Plot spectra. Data from mod.RTS_Model stores layers from deepest to shallowest, need to
    # reverse order as the spectra input requires layers from shallowest to deepest.
    plotting.plot_heatmap(spectra, [str(np.round(k)) for k in model.knots_r[::-1]],
                          title = 'Power Spectra')

    # Save spectra data
    if output_filename:
        np.savetxt(output_filename, spectra)


def correlate(
        filename1: Annotated[Path, typer.Argument(help="Path to the .sph file containing the first model data.")],
        filename2: Annotated[Path, typer.Argument(help="Path to the .sph file containing the second model data.")],    
        rmin: Annotated[float, typer.Argument(help="Minimum radius for the model.")] = _rcmb, 
        rmax: Annotated[float, typer.Argument(help="Maximum radius for the model.")] = _rmoho,
        knots: Annotated[np.ndarray, typer.Argument(help="Radial knots for the model parameterization.")] = None,
        output_filename: Annotated[str, typer.Argument(help="Path to save the computed spectra.")] = ''
        ):
    """
    Compute and plots the correlation between two geodynamic/tomographic models stored in .sph
    file, and optionally save the results. The correlation is computed at the resolution of the
    model with lower resolution.
    
    Parameters
    ----------
    filename1, filename2 : Path
        Path to the .sph files containing the model data.
    rmin : float, optional
        Minimum radius for the model, by default 3480 (Core-Mantle Boundary radius).
    rmax : float, optional
        Maximum radius for the model, by default 6346.691 (Mohorovičić discontinuity radius).
    knots : array-like, optional
        Radial knots for the model parameterization. If None, uses default knots from model files.
    output_filename : str, optional
        Path to save the correlation data as a text file, by default path is empty.
    """

    model1 = RTS_Model.from_file(filename1, rmin=rmin, rmax=rmax, knots=knots)
    model2 = RTS_Model.from_file(filename2, rmin=rmin, rmax=rmax, knots=knots)
    assert np.all(model1.knots_r == model2.knots_r), "Knot radii must be the same"
    corr = np.empty((len(model1.knots_r), model1.lmax+1))

    # loop through each layer, correlation assumes lower resolution of the two
    for ri, (coef1, coef2) in enumerate(zip(model1.coefs, model2.coefs)):
        sh_coef1 = sh.rts_to_sh(coef1)
        sh_coef2 = sh.rts_to_sh(coef2)
        _,_,corr[ri] = shtools.spectralanalysis.SHAdmitCorr(sh_coef1, sh_coef2,
                                                            normalization='ortho')

    # Plot correlation. Data from mod.RTS_Model stores layers from deepest to shallowest, need to
    # reverse order as the spectra input requires layers from shallowest to deepest.
    plotting.plot_heatmap(corr, [str(np.round(k)) for k in model1.knots_r[::-1]],
                          title = 'Correlation')

    # Save correlation data
    if output_filename:
        np.savetxt(output_filename, corr)
