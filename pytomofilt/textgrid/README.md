# `pytomofilt.textgrid`

Generate regular lon/lat grids where `z` encodes rendered text in Robinson projection space.

## Quick start

```python
from dataclasses import asdict

from pytomofilt.textgrid import TextGridConfig, generate_text_grid, save_grid_to_xyz_file

cfg = TextGridConfig(
    text="HELLO",
    center_lon=0.0,
    center_lat=0.0,
    size_deg=60.0,
    central_meridian=0.0,
    grid_spacing_deg=1.0,
    font_size=160,
)

df = generate_text_grid(cfg)  # columns: lon, lat, z
save_grid_to_xyz_file(df, "output/textgrid/hello.xyz", config_dict=asdict(cfg))
```

## Dependencies

- **Core**: `numpy`, `pandas`, `pyproj`, `pillow`
- **Default font download**: `requests`, `platformdirs`

## Font behavior

If `font_path=None`, `textgrid` will download the open-source **DejaVu Sans Bold** font on first use and cache it under your user cache directory.

To force offline behavior, set:

`PYTOMOFILT_TEXTGRID_DISABLE_FONT_DOWNLOAD=1`

In that mode, `render_text_mask(..., font_path=None)` falls back to Pillow’s built-in default font (or you can pass an explicit `font_path` to a local `.ttf`).

