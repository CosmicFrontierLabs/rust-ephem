"""Exercise CubeCL dispatch through the Python API."""

import os
from typing import Any

import numpy as np
import pytest

import rust_ephem as re

requires_gpu = pytest.mark.skipif(
    os.environ.get("RUST_EPHEM_GPU") not in {"1", "cubecl"},
    reason="requires GPU hardware",
)


@requires_gpu
def test_cubecl_masks_match_cpu(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = os.environ["RUST_EPHEM_GPU"]
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
        monkeypatch.setenv("RUST_EPHEM_GPU", backend)
        actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        np.testing.assert_array_equal(actual, expected)


@requires_gpu
def test_bright_star_catalogue_tiles(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = os.environ["RUST_EPHEM_GPU"]
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
    monkeypatch.setenv("RUST_EPHEM_GPU", backend)
    actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("selector", ["1", "cubecl"])
def test_gpu_selector_uses_cubecl_runtime(
    selector: str, ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUST_EPHEM_GPU", selector)
    monkeypatch.setenv("RUST_EPHEM_CUBECL_BACKEND", "unknown")
    rule = re.SunConstraint(min_angle=40)
    # A CPU-only build rejects the opt-in before runtime selection.
    with pytest.raises(
        RuntimeError,
        match="Unknown RUST_EPHEM_CUBECL_BACKEND|requires a build with the Cargo gpu or cubecl feature",
    ):
        rule.in_constraint_batch(
            ground_ephemeris, [0.0] * 4096, [0.0] * 4096, indices=[0] * 1024
        )
