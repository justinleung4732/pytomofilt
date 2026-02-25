"""Tests for pytomofilt.textgrid.text_renderer."""

import pytest

from pytomofilt.textgrid.text_renderer import get_default_font_path, render_text_mask


@pytest.fixture(autouse=True)
def _disable_font_download(monkeypatch):
    monkeypatch.setenv("PYTOMOFILT_TEXTGRID_DISABLE_FONT_DOWNLOAD", "1")


class TestRenderTextMask:
    def test_basic_shape(self):
        mask = render_text_mask("A", font_size=50)
        assert mask.ndim == 2
        assert mask.shape[0] > 0
        assert mask.shape[1] > 0

    def test_dtype_is_bool(self):
        mask = render_text_mask("Hi", font_size=50)
        assert mask.dtype == bool

    def test_contains_foreground(self):
        mask = render_text_mask("X", font_size=80)
        assert mask.any(), "Mask should contain at least one True pixel"

    def test_contains_background(self):
        mask = render_text_mask("I", font_size=80)
        assert not mask.all(), "Mask should contain some False pixels"

    def test_wider_for_longer_text(self):
        mask_short = render_text_mask("A", font_size=80)
        mask_long = render_text_mask("ABCDEF", font_size=80)
        assert mask_long.shape[1] > mask_short.shape[1]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            render_text_mask("")


class TestGetDefaultFontPath:
    def test_disabled_download_raises(self):
        with pytest.raises(RuntimeError):
            get_default_font_path()

