"""Tests for pytomofilt.textgrid.grid_generator."""

import numpy as np
import pytest

from pytomofilt.textgrid import TextGridConfig, generate_multi_text_grid, generate_text_grid


class TestGenerateTextGrid:
    @pytest.fixture
    def default_config(self):
        return TextGridConfig(
            text="X",
            center_lon=0.0,
            center_lat=0.0,
            size_deg=30.0,
            central_meridian=0.0,
            grid_spacing_deg=5.0,
            font_size=100,
        )

    def test_output_columns(self, default_config):
        df = generate_text_grid(default_config)
        assert list(df.columns) == ["lon", "lat", "z"]

    def test_z_values_default(self, default_config):
        df = generate_text_grid(default_config)
        unique = set(df["z"].unique())
        assert unique.issubset({0.0, 1.0})

    def test_has_nonzero_z(self, default_config):
        df = generate_text_grid(default_config)
        assert (df["z"] != 0).any(), "Grid should contain some non-zero z for 'X'"

    def test_grid_spacing(self, default_config):
        df = generate_text_grid(default_config)
        lons = np.sort(df["lon"].unique())
        diffs = np.diff(lons)
        np.testing.assert_allclose(diffs, default_config.grid_spacing_deg, atol=1e-10)

    def test_different_central_meridian(self):
        config = TextGridConfig(
            text="T",
            center_lon=90.0,
            center_lat=0.0,
            size_deg=30.0,
            central_meridian=90.0,
            grid_spacing_deg=5.0,
            font_size=100,
        )
        df = generate_text_grid(config)
        assert (df["z"] != 0).any()

    def test_custom_fill_value(self):
        config = TextGridConfig(
            text="X",
            center_lon=0.0,
            center_lat=0.0,
            size_deg=30.0,
            central_meridian=0.0,
            grid_spacing_deg=5.0,
            font_size=100,
            fill_value=42.0,
        )
        df = generate_text_grid(config)
        nonzero = df.loc[df["z"] != 0, "z"]
        assert len(nonzero) > 0
        assert (nonzero == 42.0).all()


class TestGenerateMultiTextGrid:
    def test_basic_output_columns(self):
        configs = [
            TextGridConfig(
                text="A",
                center_lon=-60.0,
                center_lat=0.0,
                size_deg=20.0,
                grid_spacing_deg=5.0,
                font_size=80,
            )
        ]
        df = generate_multi_text_grid(configs)
        assert list(df.columns) == ["lon", "lat", "z"]

    def test_two_texts_non_overlapping(self):
        c1 = TextGridConfig(
            text="A",
            center_lon=-60.0,
            center_lat=30.0,
            size_deg=20.0,
            grid_spacing_deg=5.0,
            font_size=80,
            fill_value=10.0,
        )
        c2 = TextGridConfig(
            text="B",
            center_lon=60.0,
            center_lat=-30.0,
            size_deg=20.0,
            grid_spacing_deg=5.0,
            font_size=80,
            fill_value=20.0,
        )
        df = generate_multi_text_grid([c1, c2])
        assert (df["z"] == 10.0).any(), "First text should produce z=10"
        assert (df["z"] == 20.0).any(), "Second text should produce z=20"

    def test_additive_overlap(self):
        c1 = TextGridConfig(
            text="X",
            center_lon=0.0,
            center_lat=0.0,
            size_deg=30.0,
            grid_spacing_deg=5.0,
            font_size=100,
            fill_value=5.0,
        )
        c2 = TextGridConfig(
            text="X",
            center_lon=0.0,
            center_lat=0.0,
            size_deg=30.0,
            grid_spacing_deg=5.0,
            font_size=100,
            fill_value=3.0,
        )
        df = generate_multi_text_grid([c1, c2])
        assert (df["z"] == 8.0).any(), "Overlapping text should produce additive z=8.0"

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError):
            generate_multi_text_grid([])

