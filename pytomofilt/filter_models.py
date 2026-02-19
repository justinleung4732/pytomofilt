#!/usr/bin/env python
"""
Functions to filter geodynamic models

These are used to provide a command line interface to
common filtering tasks.
"""
from pathlib import Path
from typing_extensions import Annotated
from dataclasses import dataclass

import typer

from pytomofilt import model


@dataclass
class TomographicModelFiles:
    """Input files for a tomographic model"""
    name: str
    base_path: Path
    coef_file: Path
    weights_file: Path
    evec_file: Path


def tomographic_model_from_path(input_path: Path) -> TomographicModelFiles:
    """
    tomographic_model_from_path builds and validates a tomographic model exists in input_path
    
    :param input_path: Location of tomographic files (directory)
    :type input_path: Path
    :return: Paths to each file needed for tomographic filtering
    :rtype: TomographicModelFiles
    """
    assert input_path.is_dir(), "Input path must be a directory"
    model_name = input_path.parts[-1] # Last part should be a model name
    coef_file = input_path / (model_name + ".sph")
    assert coef_file.is_file(), "Cannot see model coefficient file"
    weights_file = input_path / (model_name + ".smthp_21")
    assert weights_file.is_file(), "Cannot see model weights file"
    evec_file = input_path / (model_name + ".evc")
    assert evec_file.is_file(), "Cannot see model eigenvector file"
    return(TomographicModelFiles(
        name=model_name, base_path=input_path, coef_file=coef_file, weights_file=weights_file,
        evec_file=evec_file)
        )


def ptf_reparam_filter_files(
        tomographic_model: Annotated[Path, typer.Argument(help="Path to a directory containing a tomographic model and filter")],
        geodynamic_model: Annotated[Path, typer.Argument(help="Path to a directory containing data from a geodynamic model")]
    ):
    """
    Reparameterize and filter a geodynamic model so it can be compared with tomography
    """
    # Find the files we will need:
    tomographic_model_spec = tomographic_model_from_path(tomographic_model)
    # Build reference (e.g. tomographic) reference
    print(f"Setting up reference model for {tomographic_model_spec.name}.")
    ref_model = model.RTS_Model.from_file(tomographic_model_spec.coef_file)
    print(f"Setting up filter model for {tomographic_model_spec.name}.")
    ref_model.filter_from_file(tomographic_model_spec.evec_file, tomographic_model_spec.weights_file,
                                0.2,verbose=True)

    # Build a comparison model with the same parameterisation as the 
    # reference model
    print(f"Reading geodynamics model from {geodynamic_model}")
    comp_model = model.RTS_Model.from_directory(geodynamic_model,
            lmax=ref_model.lmax, rmin=ref_model.rmin, rmax=ref_model.rmax,
            knots=ref_model.knots_r)
    comp_model.write(geodynamic_model.parts[-1] + "_reparam.sph")
    # Apply the resolution filter from the reference model to the comparison model
    print(f"Filtering!")
    filtered_comp_model = ref_model.filter(comp_model) 
    filtered_comp_model.write(geodynamic_model.parts[-1] + "_filtered.sph")
    print(f"Done")


if __name__ == "__main__":
    # Just for easy testing - we can loose this later.
    ptf_reparam_filter_files(Path("/Volumes/Elements/LVS_2020_10_29/Code/tomofit_from_Paula/tomofilt_new_ES/utils/S12RTS"), 
                             Path("/Volumes/Elements/LVS_2020_10_29/Code/tomofit_from_Paula/tomofilt_new_ES/geodyn/examplemodel"))