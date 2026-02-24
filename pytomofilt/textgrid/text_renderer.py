"""
Render a text string to a binary numpy mask using Pillow.

If no font path is supplied, an open-source TTF (DejaVu Sans) is
automatically downloaded and cached.
"""

import os
import pathlib

import numpy as np
import requests
from platformdirs import user_cache_dir
from PIL import Image, ImageDraw, ImageFont

_DISABLE_FONT_DOWNLOAD_ENV = "PYTOMOFILT_TEXTGRID_DISABLE_FONT_DOWNLOAD"


def _is_truthy_env(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() not in {"", "0", "false", "no", "off"}


# User-writable cache directory for downloaded font(s)
_FONT_CACHE_DIR = pathlib.Path(user_cache_dir("pytomofilt", appauthor=False)) / "textgrid"

_DEFAULT_FONT_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.zip"
)
_DEFAULT_FONT_FILENAME = "DejaVuSans-Bold.ttf"


def get_default_font_path() -> pathlib.Path:
    """
    Return the path to a cached open-source TTF font file.

    Downloads DejaVu Sans Bold from GitHub on first use and caches it.
    """
    if _is_truthy_env(_DISABLE_FONT_DOWNLOAD_ENV):
        raise RuntimeError(
            "Default font download is disabled. Provide `font_path=...` or unset "
            f"{_DISABLE_FONT_DOWNLOAD_ENV}."
        )

    cached = _FONT_CACHE_DIR / _DEFAULT_FONT_FILENAME
    if cached.exists():
        return cached

    _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading default font from {_DEFAULT_FONT_URL} ...")
    import io
    import zipfile

    resp = requests.get(_DEFAULT_FONT_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(_DEFAULT_FONT_FILENAME):
                data = zf.read(name)
                tmp = cached.with_suffix(cached.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(cached)
                print(f"Font cached at {cached}")
                return cached

    raise RuntimeError(f"Could not find {_DEFAULT_FONT_FILENAME} in downloaded archive.")


def render_text_mask(
    text: str,
    font_path: str | os.PathLike | None = None,
    font_size: int = 100,
) -> np.ndarray:
    """
    Render *text* to a tight-cropped binary numpy mask.

    Returns
    -------
    mask : ndarray of bool
        2-D array where ``True`` = text foreground, ``False`` = background.
    """
    if not text:
        raise ValueError("text must be a non-empty string")

    font = None
    if font_path is None:
        try:
            font_path = get_default_font_path()
            font = ImageFont.truetype(str(font_path), font_size)
        except RuntimeError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(str(font_path), font_size)

    dummy = Image.new("1", (1, 1), color=0)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if text_w <= 0 or text_h <= 0:
        raise ValueError(f"Text '{text}' rendered with zero size.")

    img = Image.new("1", (text_w, text_h), color=1)
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, fill=0, font=font)

    arr = np.array(img, dtype=bool)
    return ~arr

