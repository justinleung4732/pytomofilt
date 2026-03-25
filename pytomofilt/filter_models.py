#!/usr/bin/env python
"""
Functions to filter geodynamic models

These are used to provide a command line interface to
common filtering tasks.
"""
from pathlib import Path
from typing_extensions import Annotated
from dataclasses import dataclass

import pooch
import typer

from pytomofilt import model


existing_models = ['S12RTS','S20RTS','S40RTS','SP12RTS']
existing_filters = pooch.create(pooch.os_cache('pytomofilt'),
                     base_url='doi:10.5281/zenodo.11212269',
                     registry = {'S12RTS.evc':'md5:9ed798063ff166f221a3dee91ef80274',
                                 'S12RTS.smthp_21':'md5:7f7ec0186599e1ec5bda8f681335a6f6',
                                 'S12RTS.sph':'md5:ba42dd1360d93f6aa2e7066be5bf658b',
                                 'S20RTS.sph':'md5:88fc71c23d4b0890460c9a50dfd11a33',
                                 'S20RTS.evc':'md5:ebd1918258bc4134205659fb453480be',
                                 'S20RTS.smthp_21':'md5:66085bb1a7331865c7d264693ca6be51',
                                 'S40RTS.sph':'md5:150649eccaa3d34c71ba4d357f1cc7e7',
                                 'S40RTS.evc':'md5:bdf5d3c33a5e827040d8d223dc9cebc4',
                                 'S40RTS.smthp_21':'md5:9bcfef203c65fb9c0647bdc94e4f6363',
                                 'SP12RTS..EP.sph':'md5:ee8e6447b5a6d1be3f0b8cf00ce75c4a',
                                 'SP12RTS..ES.sph':'md5:1956759a6b9b953394afbc4bc06bebc1',
                                 'SP12RTS.evc':'md5:aa0d63ad23271fed0387311bf6dc9706',
                                 'SP12RTS.smthp_42':'md5:da638c2c4cabecb0323b2d379e6d8e46'
                                 })


@dataclass
class TomographicModelFiles:
    """Input files for a tomographic model"""
    name: str
    base_path: Path
    coef_file: Path
    weights_file: Path
    evec_file: Path


def tomographic_model_from_name(model_name: str) -> TomographicModelFiles:
    """
    tomographic_model_from_name builds a tomographic model with the name as input
    
    :param model_name: Name of tomographic model
    :type input_path: str
    :return: Paths to each file needed for tomographic filtering
    :rtype: TomographicModelFiles
    """
    assert model_name in existing_models, f"Model name must be in one of {existing_models}"
    coef_file = existing_filters.fetch(f"{model_name}.sph")
    evec_file = existing_filters.fetch(f"{model_name}.evc")
    if "SP" in model_name:
        weights_file = existing_filters.fetch(f"{model_name}.smthp_42")
    else:
        weights_file = existing_filters.fetch(f"{model_name}.smthp_21")
    return(TomographicModelFiles(
        name=model_name, base_path=pooch.os_cache('pytomofilt'), coef_file=coef_file,
        weights_file=weights_file, evec_file=evec_file)
        )


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
        tomographic_model: Annotated[Path or str, typer.Argument(help="Name of tomographic model, or path to a directory containing a tomographic model and filter")],
        geodynamic_model: Annotated[Path, typer.Argument(help="Path to a directory containing data from a geodynamic model")]
    ):
    """
    Reparameterize and filter a geodynamic model so it can be compared with tomography
    """
    # Find the files we will need:
    if tomographic_model in existing_models:
        tomographic_model_spec = tomographic_model_from_name(tomographic_model)
    else:
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