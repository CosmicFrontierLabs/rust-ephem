"""Exercise native hardware dispatch through the Python API."""

import os
from typing import Any

import numpy as np
import pytest

import rust_ephem as re


@pytest.mark.skipif(
    os.environ.get("RUST_EPHEM_GPU") != "1", reason="requires GPU hardware"
)
def test_native_gpu_masks_match_cpu(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cross the real dispatch threshold with duplicate and reordered epochs.
    indices = list(range(7, -1, -1)) * 128
    ra = np.linspace(0, 360, 4096, endpoint=False).tolist()
    dec = np.linspace(-90, 90, 4096).tolist()
    for rule in [
        re.SunConstraint(min_angle=40, max_angle=140),
        re.MoonConstraint(min_angle=25),
        re.MoonLimbConstraint(min_angle=30, max_angle=150),
        re.EarthLimbConstraint(min_angle=10, max_angle=75),
        re.OrbitRamConstraint(min_angle=30, max_angle=100),
        re.OrbitPoleConstraint(min_angle=30),
        re.OrbitPoleConstraint(min_angle=20, max_angle=70, earth_limb_pole=True),
        re.BodyConstraint(body="Moon", min_angle=35, max_angle=110),
    ]:
        monkeypatch.delenv("RUST_EPHEM_GPU")
        expected = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        monkeypatch.setenv("RUST_EPHEM_GPU", "1")
        actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("RUST_EPHEM_GPU") != "1", reason="requires GPU hardware"
)
def test_bright_star_catalogue_tiles(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Include hits unique to either star tile, ties near the cone boundary,
    # and clear targets. The result repeats across the selected epochs.
    stars = [(0.0, 0.0)] * 32_768 + [(90.0, 0.0)]
    rule = re.BrightStarConstraint(stars=stars, fov_radius=0.05)
    ra = [0.0, 90.0, 90.049999, 90.050001, 180.0] * 420
    dec = [0.0] * len(ra)
    indices = [3, 1, 3]
    monkeypatch.delenv("RUST_EPHEM_GPU")
    expected = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
    np.testing.assert_array_equal(expected[:5, 0], [True, True, True, False, False])
    monkeypatch.setenv("RUST_EPHEM_GPU", "1")
    actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
    np.testing.assert_array_equal(actual, expected)
