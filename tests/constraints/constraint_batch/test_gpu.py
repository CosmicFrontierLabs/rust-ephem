"""Exercise CubeCL dispatch through the Python API."""

import os
from typing import Any

import numpy as np
import numpy.typing as npt
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
        re.AirmassConstraint(min_airmass=1.1, max_airmass=2.5),
        re.AirmassConstraint(max_airmass=40),
        re.MoonPhaseConstraint(
            max_illumination=1,
            min_distance=25,
            max_distance=140,
            enforce_when_below_horizon=True,
        ),
        re.SolarRollConstraint(roll_deg=37, tolerance_deg=35, panel_normal=(0, 1, 2)),
        re.SolarRollConstraint(roll_deg=-270, tolerance_deg=115),
        re.AltAzConstraint(
            min_altitude=10, max_altitude=70, min_azimuth=310, max_azimuth=40
        ),
        re.AltAzConstraint(min_azimuth=20, max_azimuth=310),
        re.AltAzConstraint(polygon=[(10, 20), (25, 80), (70, 35)]),
        re.BodyConstraint(
            body="Moon", fov_polygon=[(-20, -5), (10, -10), (5, 15)], roll_deg=45
        ),
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


@requires_gpu
@pytest.mark.parametrize(
    "rule",
    [
        re.SunConstraint(min_angle=40, max_angle=140),
        re.MoonLimbConstraint(min_angle=45),
        re.EarthLimbConstraint(min_angle=25, max_angle=85),
    ],
)
def test_apparent_gpu_masks(
    rule: Any, ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ra = np.linspace(0, 360, 4096, endpoint=False).tolist()
    dec = np.linspace(-90, 90, 4096).tolist()
    indices = [1, 4, 0, 3] * 256
    backend = os.environ["RUST_EPHEM_GPU"]
    monkeypatch.delenv("RUST_EPHEM_GPU")
    expected = rule.in_constraint_batch(
        ground_ephemeris, ra, dec, indices=indices, apparent=True
    )
    monkeypatch.setenv("RUST_EPHEM_GPU", backend)
    actual = rule.in_constraint_batch(
        ground_ephemeris, ra, dec, indices=indices, apparent=True
    )
    np.testing.assert_array_equal(actual, expected)


@requires_gpu
def test_nearest_star_values_across_tiles(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # f32 cannot distinguish the two close candidates near RA=0. The
    # nearest one is in the last tile, and a target at 180 has a distant tie.
    stars = [(60.0, 0.0)] * 32767 + [(0.0002, 0.0), (0.0001, 0.0)]
    rule = re.BrightStarConstraint(stars=stars, fov_radius=1)
    ra = [0.0, 60.0, 180.0] * 683
    dec = [0.0] * len(ra)

    def nearest() -> npt.NDArray[np.float64]:
        grid = rule.evaluate_grid(ground_ephemeris, ra, dec, indices=[0])
        return grid.constraint_values["nearest_bright_star_angle_deg"]

    backend = os.environ["RUST_EPHEM_GPU"]
    monkeypatch.delenv("RUST_EPHEM_GPU")
    expected = nearest()
    monkeypatch.setenv("RUST_EPHEM_GPU", backend)
    np.testing.assert_array_equal(nearest(), expected)


@requires_gpu
@pytest.mark.parametrize("tolerance", [0.0, 90.0, 135.0, 179.999999])
def test_solar_roll_wedge_matches_native(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch, tolerance: float
) -> None:
    rule = re.SolarRollConstraint(
        roll_deg=-23, tolerance_deg=tolerance, panel_normal=(1, -2, 3)
    )
    # 129 epochs exercises a partial packed word and crosses the trig dispatch threshold.
    indices = [0, 4, 1] * 43
    ra = np.linspace(0, 360, 1025).tolist()
    dec = np.linspace(-90, 90, 1025).tolist()
    monkeypatch.setenv("RUST_EPHEM_GPU", "0")
    expected = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
    monkeypatch.setenv("RUST_EPHEM_GPU", "1")
    actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
    np.testing.assert_array_equal(actual, expected)


@requires_gpu
def test_airmass_tie(ground_ephemeris: Any) -> None:
    rule = re.AirmassConstraint(max_airmass=2.5)
    value = rule.evaluate_grid(
        ground_ephemeris, [0.0], [0.0], indices=[0]
    ).constraint_values["airmass"][0, 0]
    assert 1 < value < 40
    ra = [0.0] * 4096
    indices = [0] * 1024
    for bound, expected in [(value, False), (np.nextafter(value, 0.0), True)]:
        result = re.AirmassConstraint(max_airmass=float(bound)).in_constraint_batch(
            ground_ephemeris, ra, ra, indices=indices
        )
        assert np.all(result == expected)


@pytest.mark.parametrize(
    "rule",
    [
        re.AirmassConstraint(max_airmass=2),
        re.AltAzConstraint(min_altitude=20),
        re.AltAzConstraint(polygon=[(10, 20), (25, 80), (70, 35)]),
        re.SolarRollConstraint(roll_deg=30, tolerance_deg=20),
        re.BodyConstraint(
            body="Moon", fov_polygon=[(-2, -2), (2, -2), (0, 2)], roll_deg=20
        ),
    ],
)
def test_trigonometric_dispatch_threshold(
    rule: Any, ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUST_EPHEM_GPU", "1")
    monkeypatch.setenv("RUST_EPHEM_CUBECL_BACKEND", "unknown")
    ra = [10.0] * 1024
    dec = [20.0] * 1024
    rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=[0] * 127)
    with pytest.raises(
        RuntimeError, match="Unknown RUST_EPHEM_CUBECL_BACKEND|requires a build"
    ):
        rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=[0] * 128)
