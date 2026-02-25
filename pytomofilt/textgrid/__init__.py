"""
textgrid — Generate lat/lon grids that render text in Robinson projection.
"""

from .projection import forward, inverse, get_robinson_transformer
from .text_renderer import render_text_mask, get_default_font_path
from .grid_generator import TextGridConfig, generate_text_grid, generate_multi_text_grid
from .io_utils import save_grid, save_grid_to_xyz_file, load_grid

__all__ = [
    "forward",
    "inverse",
    "get_robinson_transformer",
    "render_text_mask",
    "get_default_font_path",
    "TextGridConfig",
    "generate_text_grid",
    "generate_multi_text_grid",
    "save_grid",
    "save_grid_to_xyz_file",
    "load_grid",
]
