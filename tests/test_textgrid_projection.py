"""Tests for pytomofilt.textgrid.projection."""

import numpy as np
import pytest

from pytomofilt.textgrid.projection import forward, get_robinson_transformer, inverse


class TestForwardInverseRoundTrip:
    @pytest.mark.parametrize("central_lon", [0.0, 90.0, -120.0, 180.0])
    def test_roundtrip_scalar(self, central_lon):
        lon, lat = 30.0, 45.0
        x, y = forward(lon, lat, central_lon)
        lon2, lat2 = inverse(x, y, central_lon)
        assert abs(lon2 - lon) < 1e-6
        assert abs(lat2 - lat) < 1e-6

    def test_roundtrip_array(self):
        lons = np.array([-179, -90, 0, 90, 179])
        lats = np.array([-60, -30, 0, 30, 60])
        x, y = forward(lons, lats, 0.0)
        lon2, lat2 = inverse(x, y, 0.0)
        np.testing.assert_allclose(lon2, lons, atol=1e-4)
        np.testing.assert_allclose(lat2, lats, atol=1e-4)


class TestEdgeCases:
    def test_equator_origin(self):
        x, y = forward(0.0, 0.0, 0.0)
        assert abs(x) < 1e-3
        assert abs(y) < 1e-3

    def test_north_pole(self):
        x, y = forward(0.0, 90.0, 0.0)
        assert abs(x) < 1e-3
        assert y > 0

    def test_south_pole(self):
        x, y = forward(0.0, -90.0, 0.0)
        assert abs(x) < 1e-3
        assert y < 0


class TestTransformerFactory:
    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            get_robinson_transformer(0, direction="sideways")

