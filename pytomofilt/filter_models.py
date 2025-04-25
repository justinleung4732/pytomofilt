#!/usr/bin/env python
"""
Command line based tool to filter geodynamic models
"""
from pathlib import Path

from pytomofilt import model

# File names and paths - for now hard coded to AMW's machine
# but we'll need a little module to handle download and so on
S12RTS_model_spec = {"path": Path('/Volumes/Elements/LVS_2020_10_29/Code/tomofit_from_Paula/tomofilt_new_ES/utils/S12RTS'),
                     "coef_file": "S12RTS.sph",
                     "weights_file": "S12RTS.smthp_21",
                     "evec_file": "S12RTS.evc"}

def main():
    # Handle arguments and file globbing
    # Call filter models
    filter_models("S12RTS", ["/Volumes/Elements/LVS_2020_10_29/Code/tomofit_from_Paula/tomofilt_new_ES/geodyn/examplemodel"])


def filter_models(tomographic_model_spec, model_directories):
    """

    """
    # Build reference (e.g. tomographic) reference
    if tomographic_model_spec == "S12RTS":
        print("Setting up reference model and filter using S12RTS.")
        ref_model = model.RTS_Model.from_file(S12RTS_model_spec["path"]/S12RTS_model_spec["coef_file"])
        ref_model.filter_from_file(S12RTS_model_spec["path"]/S12RTS_model_spec["evec_file"],
                                    S12RTS_model_spec["path"]/S12RTS_model_spec["weights_file"], 0.2,
                                    verbose=True)
    else:
        # How else do we want to provide this information
        raise NotImplementedError
    
    # Loop over directories containing models for comparison (e.g. geodynamic models)
    for comp_model_directory in model_directories:
        # Build a comparison model with the same parameterisation as the 
        # reference model
        comp_model = model.RTS_Model.from_directory(comp_model_directory,
                lmax=ref_model.lmax, rmin=ref_model.rmin, rmax=ref_model.rmax,
                knots=ref_model.knots_r)
        # Apply the resolution filter from the reference model to the comparison model
        filtered_comp_model = ref_model.filter(comp_model) 
        # Calculate correlation between models
        correlation = ref_model.correlate(filtered_comp_model)
        # Output results
        print(f"Correlation between S12RTS and {comp_model_directory} is {correlation}")


if __name__ == "__main__":
    main()