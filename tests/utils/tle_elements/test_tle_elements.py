from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from rust_ephem.tle import (
    WGS72_EARTH_MU_M3_S2,
    TLERecord,
    true_anomaly_from_mean_anomaly,
)

# ISS TLE from tests/conftest.py
_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.fixture
def iss_tle() -> TLERecord:
    return TLERecord(
        line1=_TLE1,
        line2=_TLE2,
        epoch=datetime(2008, 9, 20, 12, 25, 40, tzinfo=timezone.utc),
    )


class TestTrueAnomalyFromMeanAnomaly:
    def test_circular_orbit_wraps_input(self) -> None:
        assert true_anomaly_from_mean_anomaly(725.0, 0.0) == pytest.approx(5.0)

    def test_circular_orbit_true_equals_mean(self) -> None:
        assert true_anomaly_from_mean_anomaly(90.0, 0.0) == pytest.approx(90.0)

    def test_rejects_parabolic_eccentricity(self) -> None:
        with pytest.raises(ValueError, match="Only elliptical TLE eccentricities"):
            true_anomaly_from_mean_anomaly(10.0, 1.0)

    def test_rejects_hyperbolic_eccentricity(self) -> None:
        with pytest.raises(ValueError, match="Only elliptical TLE eccentricities"):
            true_anomaly_from_mean_anomaly(10.0, 1.5)


class TestClassicalElements:
    def test_inclination(self, iss_tle: TLERecord) -> None:
        assert iss_tle.classical_elements()["Inclination_deg"] == pytest.approx(51.6416)

    def test_raan(self, iss_tle: TLERecord) -> None:
        assert iss_tle.classical_elements()["RightAscension_deg"] == pytest.approx(
            247.4627
        )

    def test_eccentricity(self, iss_tle: TLERecord) -> None:
        assert iss_tle.classical_elements()["Eccentricity"] == pytest.approx(0.0006703)

    def test_arg_periapsis(self, iss_tle: TLERecord) -> None:
        assert iss_tle.classical_elements()["ArgPeriapsis_deg"] == pytest.approx(
            130.5360
        )

    def test_mean_anomaly(self, iss_tle: TLERecord) -> None:
        assert iss_tle.classical_elements()["MeanAnomaly_deg"] == pytest.approx(
            325.0288
        )

    def test_mean_motion_uses_column_slice_not_split(self, iss_tle: TLERecord) -> None:
        # line2[52:63] = "15.72125391"; .split()[7] = "15.72125391563537" (wrong)
        assert iss_tle.classical_elements()["MeanMotion_rev_per_day"] == pytest.approx(
            15.72125391
        )

    def test_semimajor_axis(self, iss_tle: TLERecord) -> None:
        n = 15.72125391 * 2.0 * math.pi / 86400.0
        expected_a = (WGS72_EARTH_MU_M3_S2 / n**2) ** (1.0 / 3.0)
        assert iss_tle.classical_elements()["SemimajorAxis_m"] == pytest.approx(
            expected_a, rel=1e-9
        )

    def test_gravitational_parameter_default(self, iss_tle: TLERecord) -> None:
        assert (
            iss_tle.classical_elements()["GravitationalParameter_m3_s2"]
            == WGS72_EARTH_MU_M3_S2
        )

    def test_gravitational_parameter_override(self, iss_tle: TLERecord) -> None:
        custom_mu = 3.986004418e14
        elements = iss_tle.classical_elements(mu_m3_s2=custom_mu)
        assert elements["GravitationalParameter_m3_s2"] == custom_mu
        n = 15.72125391 * 2.0 * math.pi / 86400.0
        expected_a = (custom_mu / n**2) ** (1.0 / 3.0)
        assert elements["SemimajorAxis_m"] == pytest.approx(expected_a, rel=1e-9)

    def test_true_anomaly_close_to_mean_for_low_eccentricity(
        self, iss_tle: TLERecord
    ) -> None:
        elements = iss_tle.classical_elements()
        # For e ≈ 0.0007 the max difference between ν and M is ~0.08°
        assert abs(
            elements["TrueAnomaly_deg"] - elements["MeanAnomaly_deg"]
        ) < pytest.approx(0.1)
