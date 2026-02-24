"""
Build a regular lat/lon grid and sample a text mask in Robinson projection space.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import projection
from . import text_renderer


@dataclass
class TextGridConfig:
    """
    Configuration for text-grid generation.

    Parameters
    ----------
    text : str
        The text string to embed in the grid.
    center_lon : float
        Longitude (degrees) of the centre of the text placement.
    center_lat : float
        Latitude (degrees) of the centre of the text placement.
    size_deg : float
        Width of the text bounding box in degrees of longitude. The height
        is derived automatically from the aspect ratio of the rendered text.
    central_meridian : float
        Central meridian of the Robinson projection (degrees). When data
        is plotted with this central meridian the text will appear undistorted.
    grid_spacing_deg : float
        Grid spacing in degrees for both longitude and latitude.
    font_path : str, Path or None
        Path to a ``.ttf`` font file. ``None`` = auto-download an open-source font.
    font_size : int
        Font size passed to the text renderer (controls rasterisation quality).
    fill_value : float
        The z-value assigned to grid points that fall inside the rendered
        text. Points outside the text remain 0. Default is ``1.0``.
    """

    text: str = "HELLO"
    center_lon: float = 0.0
    center_lat: float = 0.0
    size_deg: float = 60.0
    central_meridian: float = 0.0
    grid_spacing_deg: float = 1.0
    font_path: Optional[str] = None
    font_size: int = 200
    fill_value: float = 1.0


def generate_text_grid(config: TextGridConfig) -> pd.DataFrame:
    """
    Generate a lat/lon/z grid where z encodes text in Robinson projection space.

    Returns
    -------
    pd.DataFrame
        Columns ``lon``, ``lat``, ``z``.
    """
    mask = text_renderer.render_text_mask(
        config.text,
        font_path=config.font_path,
        font_size=config.font_size,
    )
    mask_h, mask_w = mask.shape

    cx, cy = projection.forward(config.center_lon, config.center_lat, config.central_meridian)

    left_lon = config.center_lon - config.size_deg / 2
    right_lon = config.center_lon + config.size_deg / 2
    x_left, _ = projection.forward(left_lon, config.center_lat, config.central_meridian)
    x_right, _ = projection.forward(right_lon, config.center_lat, config.central_meridian)
    bbox_width_m = abs(x_right - x_left)

    aspect = mask_h / mask_w
    bbox_height_m = bbox_width_m * aspect

    x_min = cx - bbox_width_m / 2
    y_max = cy + bbox_height_m / 2

    lons = np.arange(-180, 180, config.grid_spacing_deg)
    lats = np.arange(-90, 90 + config.grid_spacing_deg / 2, config.grid_spacing_deg)
    lats = lats[lats <= 90]

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    x_all, y_all = projection.forward(lon_flat, lat_flat, config.central_meridian)

    z = np.zeros(len(lon_flat), dtype=np.float64)

    px = ((x_all - x_min) / bbox_width_m * mask_w).astype(np.float64)
    py = ((y_max - y_all) / bbox_height_m * mask_h).astype(np.float64)

    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)

    inside = (ix >= 0) & (ix < mask_w) & (iy >= 0) & (iy < mask_h)
    z[inside] = np.where(mask[iy[inside], ix[inside]], config.fill_value, 0.0)

    return pd.DataFrame({"lon": lon_flat, "lat": lat_flat, "z": z})


def _sample_text_to_z(config: TextGridConfig, lon_flat: np.ndarray, lat_flat: np.ndarray) -> np.ndarray:
    mask = text_renderer.render_text_mask(
        config.text,
        font_path=config.font_path,
        font_size=config.font_size,
    )
    mask_h, mask_w = mask.shape

    cx, cy = projection.forward(config.center_lon, config.center_lat, config.central_meridian)

    left_lon = config.center_lon - config.size_deg / 2
    right_lon = config.center_lon + config.size_deg / 2
    x_left, _ = projection.forward(left_lon, config.center_lat, config.central_meridian)
    x_right, _ = projection.forward(right_lon, config.center_lat, config.central_meridian)
    bbox_width_m = abs(x_right - x_left)

    aspect = mask_h / mask_w
    bbox_height_m = bbox_width_m * aspect

    x_min = cx - bbox_width_m / 2
    y_max = cy + bbox_height_m / 2

    x_all, y_all = projection.forward(lon_flat, lat_flat, config.central_meridian)

    z = np.zeros(len(lon_flat), dtype=np.float64)

    px = ((x_all - x_min) / bbox_width_m * mask_w).astype(np.float64)
    py = ((y_max - y_all) / bbox_height_m * mask_h).astype(np.float64)

    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)

    inside = (ix >= 0) & (ix < mask_w) & (iy >= 0) & (iy < mask_h)
    z[inside] = np.where(mask[iy[inside], ix[inside]], config.fill_value, 0.0)
    return z


def generate_multi_text_grid(configs: list[TextGridConfig]) -> pd.DataFrame:
    """
    Generate a single lat/lon/z grid with multiple text objects composited.

    Where multiple texts overlap, their fill values are added.
    """
    if not configs:
        raise ValueError("configs must contain at least one TextGridConfig")

    spacing = configs[0].grid_spacing_deg

    lons = np.arange(-180, 180, spacing)
    lats = np.arange(-90, 90 + spacing / 2, spacing)
    lats = lats[lats <= 90]

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    z_total = np.zeros(len(lon_flat), dtype=np.float64)
    for cfg in configs:
        z_total += _sample_text_to_z(cfg, lon_flat, lat_flat)

    return pd.DataFrame({"lon": lon_flat, "lat": lat_flat, "z": z_total})

