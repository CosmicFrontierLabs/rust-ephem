import os
from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt

from rust_ephem import GroundEphemeris

# Tolerance constants
ALT_P95_TOL = 0.32
ALT_MAX_TOL = 0.35
AZ_P95_TOL = 0.7
AZ_MAX_TOL = 0.8


def _minimal_angle_diff_deg(
    a_deg: npt.NDArray[np.float64], b_deg: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d


def _compute_altaz(
    ephem: GroundEphemeris, ra_deg: float, dec_deg: float, fast: bool
) -> npt.NDArray[np.float64]:
    if fast:
        os.environ["RUST_EPHEM_FAST_ALTAZ"] = "1"
    else:
        os.environ.pop("RUST_EPHEM_FAST_ALTAZ", None)
    out = ephem.radec_to_altaz(ra_deg, dec_deg)
    return np.asarray(out)


class TestMaunaKea:
    site = {"lat": 19.826, "lon": -155.47, "h": 4205.0}
    targets = [
        (278.62, 31.34),
        (158.00, 34.33),
        (309.10, -44.63),
        (251.05, -5.95),
        (33.90, -15.50),
    ]

    def _compute_errors(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # 1 hour window, 5s step for modest runtime
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = begin + timedelta(hours=1)

        eph = GroundEphemeris(
            self.site["lat"],
            self.site["lon"],
            self.site["h"],
            begin,
            end,
            5,
            polar_motion=True,
        )

        # Check across targets
        alt_errs = []
        az_errs = []

        for ra, dec in self.targets:
            altaz_prec = _compute_altaz(eph, ra, dec, fast=False)
            altaz_fast = _compute_altaz(eph, ra, dec, fast=True)

            alt_prec, az_prec = altaz_prec[:, 0], altaz_prec[:, 1]
            alt_fast, az_fast = altaz_fast[:, 0], altaz_fast[:, 1]

            alt_errs.append(np.abs(alt_fast - alt_prec))
            az_errs.append(np.abs(_minimal_angle_diff_deg(az_fast, az_prec)))

        alt_err = np.concatenate(alt_errs)
        az_err = np.concatenate(az_errs)
        return alt_err, az_err

    def test_alt_p95(self) -> None:
        alt_err, _ = self._compute_errors()
        # Altitude: expect ~<= 0.3 deg typical; allow small margin
        assert np.percentile(alt_err, 95) <= ALT_P95_TOL, (
            f"alt p95 too large: {np.percentile(alt_err, 95):.4f} deg"
        )

    def test_alt_max(self) -> None:
        alt_err, _ = self._compute_errors()
        assert alt_err.max() <= ALT_MAX_TOL, (
            f"alt max too large: {alt_err.max():.4f} deg"
        )

    def test_az_p95(self) -> None:
        _, az_err = self._compute_errors()
        # Azimuth: with corrected convention and GMST approximation, allow ~<=0.7 deg p95 and <=0.8 deg max
        assert np.percentile(az_err, 95) <= AZ_P95_TOL, (
            f"az p95 too large: {np.percentile(az_err, 95):.4f} deg"
        )

    def test_az_max(self) -> None:
        _, az_err = self._compute_errors()
        assert az_err.max() <= AZ_MAX_TOL, f"az max too large: {az_err.max():.4f} deg"


class TestParanal:
    site = {"lat": -24.6272, "lon": -70.4042, "h": 2635.0}
    targets = [
        (12.3, -30.0),
        (120.0, 20.0),
        (230.0, -10.0),
        (300.0, 10.0),
        (350.0, 45.0),
    ]

    def _compute_errors(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # 1 hour window, 5s step for modest runtime
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = begin + timedelta(hours=1)

        eph = GroundEphemeris(
            self.site["lat"],
            self.site["lon"],
            self.site["h"],
            begin,
            end,
            5,
            polar_motion=True,
        )

        # Check across targets
        alt_errs = []
        az_errs = []

        for ra, dec in self.targets:
            altaz_prec = _compute_altaz(eph, ra, dec, fast=False)
            altaz_fast = _compute_altaz(eph, ra, dec, fast=True)

            alt_prec, az_prec = altaz_prec[:, 0], altaz_prec[:, 1]
            alt_fast, az_fast = altaz_fast[:, 0], altaz_fast[:, 1]

            alt_errs.append(np.abs(alt_fast - alt_prec))
            az_errs.append(np.abs(_minimal_angle_diff_deg(az_fast, az_prec)))

        alt_err = np.concatenate(alt_errs)
        az_err = np.concatenate(az_errs)
        return alt_err, az_err

    def test_alt_p95(self) -> None:
        alt_err, _ = self._compute_errors()
        # Altitude: expect ~<= 0.3 deg typical; allow small margin
        assert np.percentile(alt_err, 95) <= ALT_P95_TOL, (
            f"alt p95 too large: {np.percentile(alt_err, 95):.4f} deg"
        )

    def test_alt_max(self) -> None:
        alt_err, _ = self._compute_errors()
        assert alt_err.max() <= ALT_MAX_TOL, (
            f"alt max too large: {alt_err.max():.4f} deg"
        )

    def test_az_p95(self) -> None:
        _, az_err = self._compute_errors()
        # Azimuth: with corrected convention and GMST approximation, allow ~<=0.7 deg p95 and <=0.8 deg max
        assert np.percentile(az_err, 95) <= AZ_P95_TOL, (
            f"az p95 too large: {np.percentile(az_err, 95):.4f} deg"
        )

    def test_az_max(self) -> None:
        _, az_err = self._compute_errors()
        assert az_err.max() <= AZ_MAX_TOL, f"az max too large: {az_err.max():.4f} deg"


class TestGreenwich:
    site = {"lat": 51.4769, "lon": 0.0005, "h": 46.0}
    targets = [
        (45.0, 0.0),
        (90.0, 20.0),
        (150.0, 40.0),
        (210.0, -20.0),
        (330.0, 50.0),
    ]

    def _compute_errors(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # 1 hour window, 5s step for modest runtime
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = begin + timedelta(hours=1)

        eph = GroundEphemeris(
            self.site["lat"],
            self.site["lon"],
            self.site["h"],
            begin,
            end,
            5,
            polar_motion=True,
        )

        # Check across targets
        alt_errs = []
        az_errs = []

        for ra, dec in self.targets:
            altaz_prec = _compute_altaz(eph, ra, dec, fast=False)
            altaz_fast = _compute_altaz(eph, ra, dec, fast=True)

            alt_prec, az_prec = altaz_prec[:, 0], altaz_prec[:, 1]
            alt_fast, az_fast = altaz_fast[:, 0], altaz_fast[:, 1]

            alt_errs.append(np.abs(alt_fast - alt_prec))
            az_errs.append(np.abs(_minimal_angle_diff_deg(az_fast, az_prec)))

        alt_err = np.concatenate(alt_errs)
        az_err = np.concatenate(az_errs)
        return alt_err, az_err

    def test_alt_p95(self) -> None:
        alt_err, _ = self._compute_errors()
        # Altitude: expect ~<= 0.3 deg typical; allow small margin
        assert np.percentile(alt_err, 95) <= ALT_P95_TOL, (
            f"alt p95 too large: {np.percentile(alt_err, 95):.4f} deg"
        )

    def test_alt_max(self) -> None:
        alt_err, _ = self._compute_errors()
        assert alt_err.max() <= ALT_MAX_TOL, (
            f"alt max too large: {alt_err.max():.4f} deg"
        )

    def test_az_p95(self) -> None:
        _, az_err = self._compute_errors()
        # Azimuth: with corrected convention and GMST approximation, allow ~<=0.7 deg p95 and <=0.8 deg max
        assert np.percentile(az_err, 95) <= AZ_P95_TOL, (
            f"az p95 too large: {np.percentile(az_err, 95):.4f} deg"
        )

    def test_az_max(self) -> None:
        _, az_err = self._compute_errors()
        assert az_err.max() <= AZ_MAX_TOL, f"az max too large: {az_err.max():.4f} deg"
