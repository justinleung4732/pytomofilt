"""
Read / write lat-lon-z grid files.
"""

import json
import pathlib
from typing import Optional

import pandas as pd


def save_grid(
    df: pd.DataFrame,
    filepath: str | pathlib.Path,
    config_dict: Optional[dict] = None,
) -> pathlib.Path:
    """
    Save a lon/lat/z DataFrame to a CSV text file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``lon``, ``lat``, ``z``.
    filepath : str or Path
        Destination file path.
    config_dict : dict, optional
        Generation parameters to store in a ``#``-prefixed JSON header line.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        if config_dict is not None:
            f.write(f"# {json.dumps(config_dict)}\n")
        df.to_csv(f, index=False)

    return filepath


def save_grid_to_xyz_file(
    df: pd.DataFrame,
    filepath: str | pathlib.Path,
    config_dict: Optional[dict] = None,
) -> pathlib.Path:
    """
    Save a lon/lat/z DataFrame to a tab-separated xyz text file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``lon``, ``lat``, ``z``.
    filepath : str or Path
        Destination file path.
    config_dict : dict, optional
        Generation parameters to store in a ``#``-prefixed JSON header line.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        if config_dict is not None:
            f.write(f"# {json.dumps(config_dict)}\n")
        df.to_csv(f, index=False, sep="\t")

    return filepath


def load_grid(filepath: str | pathlib.Path) -> tuple[pd.DataFrame, Optional[dict]]:
    """
    Load a previously saved grid file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    df : pd.DataFrame
        Columns ``lon``, ``lat``, ``z``.
    config : dict or None
        The config dict from the header, if present.
    """
    filepath = pathlib.Path(filepath)
    config = None

    with open(filepath, "r") as f:
        first_line = f.readline()
        if first_line.startswith("# "):
            try:
                config = json.loads(first_line[2:])
            except json.JSONDecodeError:
                config = None

    df = pd.read_csv(filepath, comment="#")
    return df, config
