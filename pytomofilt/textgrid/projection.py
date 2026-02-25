"""
Robinson projection helpers using pyproj.
"""

import numpy as np
from pyproj import Transformer


def get_robinson_transformer(central_lon: float = 0.0, direction: str = "forward"):
    """
    Return a pyproj Transformer for the Robinson projection.

    Parameters
    ----------
    central_lon : float
        Central meridian of the Robinson projection in degrees.
    direction : str
        ``"forward"`` for lon/lat → x/y, ``"inverse"`` for x/y → lon/lat.

    Returns
    -------
    pyproj.Transformer
    """
    proj_string = (
        f"+proj=robin +lon_0={central_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    if direction == "forward":
        return Transformer.from_proj("EPSG:4326", proj_string, always_xy=True)
    if direction == "inverse":
        return Transformer.from_proj(proj_string, "EPSG:4326", always_xy=True)
    raise ValueError(f"direction must be 'forward' or 'inverse', got '{direction}'")


def forward(lon, lat, central_lon: float = 0.0):
    """
    Project longitude / latitude → Robinson x, y (metres).

    Parameters
    ----------
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    central_lon : float
        Central meridian of the Robinson projection.

    Returns
    -------
    x, y : ndarray
        Projected coordinates in metres.
    """
    transformer = get_robinson_transformer(central_lon, direction="forward")
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def inverse(x, y, central_lon: float = 0.0):
    """
    Inverse-project Robinson x, y (metres) → longitude, latitude.

    Parameters
    ----------
    x, y : float or array-like
        Projected coordinates in metres.
    central_lon : float
        Central meridian of the Robinson projection.

    Returns
    -------
    lon, lat : ndarray
        Geographic coordinates in degrees.
    """
    transformer = get_robinson_transformer(central_lon, direction="inverse")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)
