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
    filter_models("S12RTS", "blar")


def filter_models(tomographic_model_spec, directories):
    # Build tomographic reference
    if tomographic_model_spec == "S12RTS":
        tomo_model = model.RTS_Model.from_file(S12RTS_model_spec["path"]/S12RTS_model_spec["coef_file"])
        tomo_model.filter_from_file(S12RTS_model_spec["path"]/S12RTS_model_spec["evec_file"],
                                    S12RTS_model_spec["path"]/S12RTS_model_spec["weights_file"], 0.2,
                                    verbose=True)
    else:
        # Try unpacking a tuple of filenames
        raise NotImplementedError
    # Loop over geodynamic models
    # - build model
    # - apply filter
    # - correlate
    # - output results


if __name__ == "__main__":
    main()