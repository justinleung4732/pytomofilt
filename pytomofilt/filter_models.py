#!/usr/bin/env python
"""
Command line based tool to filter geodynamic models
"""
from . import model

def main():
    # Handle arguments and file globbing
    # Call filter models
    pass


def filter_models(tomographic_model_spec, directories):
    # Build tomographic reference
    if tomographic_model_spec == "S20RTS":
        pass
    else:
        # Try unpacking a tuple of filenames
        pass
    # Loop over geodynamic models
    # - build model
    # - apply filter
    # - correlate
    # - output results


if __name__ == "__main__":
    main()