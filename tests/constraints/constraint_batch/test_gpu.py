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
    ]:
        monkeypatch.delenv("RUST_EPHEM_GPU")
        expected = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        monkeypatch.setenv("RUST_EPHEM_GPU", "1")
        actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        np.testing.assert_array_equal(actual, expected)
