"""Tests for pytomofilt.textgrid.io_utils."""

import pandas as pd

from pytomofilt.textgrid.io_utils import load_grid, save_grid


class TestSaveLoadRoundTrip:
    def _make_df(self):
        return pd.DataFrame(
            {
                "lon": [0.0, 1.0, 2.0],
                "lat": [10.0, 20.0, 30.0],
                "z": [0, 1, 0],
            }
        )

    def test_roundtrip(self, tmp_path):
        df = self._make_df()
        fpath = tmp_path / "grid.csv"
        save_grid(df, fpath)
        df2, cfg = load_grid(fpath)
        pd.testing.assert_frame_equal(df, df2)
        assert cfg is None

    def test_roundtrip_with_config(self, tmp_path):
        df = self._make_df()
        fpath = tmp_path / "grid.csv"
        config = {"text": "HI", "size_deg": 30.0}
        save_grid(df, fpath, config_dict=config)
        df2, cfg = load_grid(fpath)
        pd.testing.assert_frame_equal(df, df2)
        assert cfg == config

    def test_creates_parent_dirs(self, tmp_path):
        df = self._make_df()
        fpath = tmp_path / "sub" / "dir" / "grid.csv"
        save_grid(df, fpath)
        assert fpath.exists()

