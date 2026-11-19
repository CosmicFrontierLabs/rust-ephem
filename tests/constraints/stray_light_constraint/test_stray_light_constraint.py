"""StrayLightConstraint against an independent NumPy implementation of the
model, on a synthetic point-source transmittance table."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

import rust_ephem as re
from rust_ephem import PointSourceTransmittance, StrayLightConstraint

# Any dtype: older NumPy stubs type arange() and cross() as floating[Any].
Array = npt.NDArray[Any]

AU_KM = 1.495978707e8
RADIUS_KM = {"sun": 695700.0, "earth": 6371.0, "moon": 1737.4}
START_GRID = {"sun": (2, 4), "earth": (16, 32), "moon": (4, 8)}
BEGIN = datetime(2026, 3, 1, tzinfo=timezone.utc)
STEP = 600
N = 1201
CONVERSION = 4.0e-9


# --- fixtures ----------------------------------------------------------------


def synthetic_log10(off_axis: Array, azimuth: Array) -> Array:
    """A smooth response: steep near the axis, brighter toward azimuth 0."""
    return np.asarray(
        -3.0 - 0.05 * off_axis[:, None] + 0.5 * np.cos(np.radians(azimuth))[None, :]
    )


def make_table(symmetric: bool = True, holes: bool = True) -> PointSourceTransmittance:
    off_axis = np.concatenate([np.arange(0.0, 20.0, 2.0), np.arange(20.0, 141.0, 5.0)])
    azimuth = np.arange(0.0, 181.0, 15.0) if symmetric else np.arange(0.0, 360.0, 30.0)
    log = synthetic_log10(off_axis, azimuth)
    if not symmetric:
        log = log + 0.3 * np.sin(np.radians(azimuth))[None, :]
    if holes:
        log[-3:, 2] = np.nan  # cells the raytrace left empty
    return PointSourceTransmittance(
        off_axis_deg=off_axis.tolist(),
        azimuth_deg=azimuth.tolist(),
        log10_pst=log.tolist(),
        azimuth_symmetric=symmetric,
    )


@pytest.fixture(scope="module")
def table() -> PointSourceTransmittance:
    return make_table()


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> Any:
    """An eccentric high orbit, 30,000 to 400,000 km from the Earth, so the
    Earth ranges from a 12 degree disk to a point."""
    seconds = np.arange(N) * float(STEP)
    rate = np.sqrt(398600.4418 / 215000.0**3)
    radius = 215000.0 * (1.0 - 0.86 * np.cos(rate * seconds))
    angle = 1.3 * rate * seconds
    position = np.stack(
        [
            radius * np.cos(angle),
            0.9 * radius * np.sin(angle),
            0.43 * radius * np.sin(angle),
        ],
        axis=1,
    )
    velocity = np.gradient(position, float(STEP), axis=0)
    times = [BEGIN + timedelta(seconds=float(s)) for s in seconds]
    return re.FileEphemeris.from_arrays(times, position, velocity, step_size=STEP)


@pytest.fixture(scope="module")
def sample() -> tuple[Array, Array, npt.NDArray[np.intp]]:
    """Random pointings paired with random ephemeris rows."""
    rng = np.random.default_rng(42)
    count = 160
    return (
        rng.uniform(0.0, 360.0, count),
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count))),
        rng.integers(0, N, count),
    )


def bodies_at(ephem: Any, rows: Any) -> dict[str, Array]:
    """Vectors (km) from the observer to each body at the given rows."""
    observer = np.asarray(ephem.gcrs_pv.position)[rows]
    return {
        "sun": np.asarray(ephem.sun_pv.position)[rows] - observer,
        "earth": -observer,
        "moon": np.asarray(ephem.moon_pv.position)[rows] - observer,
    }


def evaluate(constraint: Any, ephem: Any, ra: Any, dec: Any, rows: Any) -> Any:
    times = [ephem.timestamp[int(i)] for i in np.atleast_1d(rows)]
    return constraint.evaluate_elementwise(
        ephem, np.atleast_1d(ra), np.atleast_1d(dec), times=times
    )


# --- independent reference ---------------------------------------------------


def unit(v: Array) -> Array:
    return np.asarray(v / np.linalg.norm(v, axis=-1, keepdims=True))


def angle_deg(a: Array, b: Array) -> float:
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


class ReferenceTable:
    """Bilinear lookup in log10(PST) with the documented edge rules."""

    def __init__(self, table: PointSourceTransmittance) -> None:
        self.off_axis = np.array(table.off_axis_deg)
        azimuth = np.array(table.azimuth_deg)
        log = np.array(
            [[np.nan if c is None else c for c in row] for row in table.log10_pst]
        )
        self.floor_log = float(np.nanmin(log))
        log = np.where(np.isnan(log), self.floor_log, log)
        self.symmetric = table.azimuth_symmetric
        if not self.symmetric:  # close the turn
            azimuth = np.append(azimuth, azimuth[0] + 360.0)
            log = np.concatenate([log, log[:, :1]], axis=1)
        self.azimuth, self.log = azimuth, log

    def __call__(self, polar: Array, azimuth: Array) -> Array:
        if self.symmetric:
            folded = np.abs(azimuth) % 360.0
            folded = np.where(folded > 180.0, 360.0 - folded, folded)
            az = np.clip(folded, self.azimuth[0], self.azimuth[-1])
        else:
            az = self.azimuth[0] + (azimuth - self.azimuth[0]) % 360.0
        p = np.clip(polar, self.off_axis[0], self.off_axis[-1])
        i = np.clip(
            np.searchsorted(self.off_axis, p, side="right") - 1,
            0,
            len(self.off_axis) - 2,
        )
        j = np.clip(
            np.searchsorted(self.azimuth, az, side="right") - 1,
            0,
            len(self.azimuth) - 2,
        )
        u = (p - self.off_axis[i]) / (self.off_axis[i + 1] - self.off_axis[i])
        v = (az - self.azimuth[j]) / (self.azimuth[j + 1] - self.azimuth[j])
        g = self.log
        value = (1 - u) * ((1 - v) * g[i, j] + v * g[i, j + 1]) + u * (
            (1 - v) * g[i + 1, j] + v * g[i + 1, j + 1]
        )
        return np.asarray(
            np.where(polar > self.off_axis[-1], self.floor_log, value), dtype=float
        )


def lambert_integral(x: float, alpha_deg: Array, n: int = 400) -> Array:
    """Phase law of a Lambert sphere seen from distance R / x, by a midpoint
    rule over the visible cap, normalised to the distant-observer law."""
    theta_max = np.pi / 2 if x == 0 else np.arccos(x)
    theta = (np.arange(n) + 0.5) / n * theta_max
    longitude = (np.arange(2 * n) + 0.5) / (2 * n) * np.pi
    st, ct = np.sin(theta), np.cos(theta)
    if x == 0:
        geometry = ct * st
    else:
        r2 = 1 - 2 * x * ct + x * x
        geometry = (ct - x) * (1 - x * ct) / r2 * st / r2
    weight = (theta_max / n) * (np.pi / (2 * n)) * 2 * 3 / (2 * np.pi)
    out = []
    for alpha in np.radians(np.atleast_1d(alpha_deg)):
        cos_incidence = np.clip(
            np.outer(st, np.cos(longitude)) * np.sin(alpha)
            + ct[:, None] * np.cos(alpha),
            0,
            None,
        )
        out.append(weight * np.sum(cos_incidence * geometry[:, None]))
    return np.array(out)


@lru_cache(maxsize=None)
def lambert_row(index: int) -> Array:
    """log10 of the phase law at x = index / 100 for alpha = 0, 2.5, ..., 180."""
    values = lambert_integral(index * 0.01, np.arange(0.0, 180.001, 2.5))
    return np.asarray(np.maximum(-9.0, np.log10(np.maximum(1e-300, values))))


def lambert_phase(x: float, alpha_deg: float) -> float:
    """The tabulated phase law, bilinear in log10 as the constraint reads it."""
    fx, fa = min(max(x, 0.0), 0.98) / 0.01, min(max(alpha_deg, 0.0), 180.0) / 2.5
    i, j = min(int(fx), 97), min(int(fa), 71)
    u, v = fx - i, fa - j
    low, high = lambert_row(i), lambert_row(i + 1)
    return float(
        10.0
        ** (
            (1 - u) * ((1 - v) * low[j] + v * low[j + 1])
            + u * ((1 - v) * high[j] + v * high[j + 1])
        )
    )


ZODI_TABLE = np.array(
    [
        [np.nan, np.nan, np.nan, np.nan, 22.6, 23.0, 23.3],
        [np.nan, np.nan, np.nan, np.nan, 22.6, 23.1, 23.3],
        [np.nan, np.nan, np.nan, 22.3, 22.7, 23.1, 23.3],
        [np.nan, np.nan, 22.1, 22.5, 22.9, 23.1, 23.3],
        [21.3, 21.9, 22.4, 22.7, 23.0, 23.2, 23.3],
        [21.7, 22.2, 22.6, 22.9, 23.1, 23.2, 23.3],
        [22.0, 22.3, 22.7, 23.0, 23.2, 23.3, 23.3],
        [22.2, 22.5, 22.9, 23.1, 23.3, 23.3, 23.3],
        [22.4, 22.6, 22.9, 23.2, 23.3, 23.3, 23.3],
        [22.4, 22.6, 22.9, 23.2, 23.3, 23.4, 23.3],
        [22.4, 22.6, 22.9, 23.1, 23.3, 23.4, 23.3],
        [22.3, 22.5, 22.8, 23.0, 23.2, 23.4, 23.3],
        [22.1, 22.4, 22.7, 23.0, 23.2, 23.4, 23.3],
    ]
)


def ecliptic(v: Array) -> tuple[float, float]:
    e = np.radians(23.4392911)
    x, y, z = (
        v[0],
        np.cos(e) * v[1] + np.sin(e) * v[2],
        -np.sin(e) * v[1] + np.cos(e) * v[2],
    )
    return float(np.degrees(np.arctan2(y, x)) % 360.0), float(
        np.degrees(np.arcsin(z / np.linalg.norm(v)))
    )


def zodi_table(boresight: Array, sun: Array) -> float:
    longitude, latitude = ecliptic(boresight)
    from_sun = (longitude - ecliptic(sun)[0]) % 360.0
    fx, fy = abs((from_sun + 180.0) % 360.0 - 180.0) / 15.0, abs(latitude) / 15.0
    i, j = min(int(fx), 11), min(int(fy), 5)
    u, v = fx - i, fy - j
    t = ZODI_TABLE
    return float(
        (1 - u) * ((1 - v) * t[i, j] + v * t[i, j + 1])
        + u * ((1 - v) * t[i + 1, j] + v * t[i + 1, j + 1])
    )


class Reference:
    """The model, one pointing at a time."""

    def __init__(self, constraint: StrayLightConstraint) -> None:
        self.c = constraint
        self.pst = ReferenceTable(constraint.pst)

    def frame(
        self, b: Array, sun: Array, roll: float | None
    ) -> tuple[Array, Array, float]:
        """Table axes (y, x) and the spacecraft roll they correspond to.

        Roll zero puts body +Y to the east; roll turns it toward north. With no
        roll given, body +Y is on the Sun's side, turned by roll_offset_deg.
        The table's azimuth-zero axis is body +Y turned by sun_azimuth_deg."""
        east = unit(np.cross([0.0, 0.0, 1.0], b))
        north = np.cross(b, east)
        if roll is None:
            s = unit(sun)
            sun_side = unit(s - (s @ b) * b)
            offset = np.radians(self.c.roll_offset_deg)
            body_y = np.cos(offset) * sun_side + np.sin(offset) * np.cross(b, sun_side)
            roll = float(np.degrees(np.arctan2(body_y @ north, body_y @ east)))
        angle = np.radians(roll)
        body_y = np.cos(angle) * east + np.sin(angle) * north
        q = np.radians(self.c.sun_azimuth_deg)
        y = np.cos(q) * body_y + np.sin(q) * np.cross(b, body_y)
        return y, np.cross(y, b), roll % 360.0

    def level(
        self,
        b: Array,
        y: Array,
        x: Array,
        centre: Array,
        radius: float,
        sun: Array,
        n_r: int,
        n_az: int,
        lambert: bool,
    ) -> tuple[float, float, float]:
        d = float(np.linalg.norm(centre))
        e = centre / d
        rho = np.arcsin(radius / d)
        ref = np.array([0.0, 0.0, 1.0]) if abs(e[2]) < 0.95 else np.array([0.0, 1.0, 0])
        u = unit(np.cross(ref, e))
        w = np.cross(e, u)
        gamma, beta = (
            a.ravel()
            for a in np.meshgrid(
                rho * (np.arange(n_r) + 0.5) / n_r,
                (np.arange(n_az) + 0.5) / n_az * 2 * np.pi,
                indexing="ij",
            )
        )
        p = np.cos(gamma)[:, None] * e + np.sin(gamma)[:, None] * (
            np.cos(beta)[:, None] * u + np.sin(beta)[:, None] * w
        )
        polar = np.degrees(np.arccos(np.clip(p @ b, -1, 1)))
        value = 10.0 ** self.pst(polar, np.degrees(np.arctan2(p @ x, p @ y)))
        flat = np.sin(gamma)
        weight = flat
        if lambert:
            reach = d * np.cos(gamma)
            hit = reach - np.sqrt(np.clip(reach**2 - (d * d - radius**2), 0, None))
            surface = hit[:, None] * p
            cos_incidence = np.sum(unit(surface - centre) * unit(sun - surface), axis=1)
            weight = flat * np.clip(cos_incidence, 0, None)
        chosen = weight if weight.sum() > 0 else flat
        return (
            float(np.sum(chosen * value) / chosen.sum()),
            float(np.mean(weight > 0)),
            float((value.max() - value.min()) / value.mean()),
        )

    def body(
        self, name: str, b: Array, y: Array, x: Array, centre: Array, sun: Array
    ) -> tuple[float, bool, dict[str, float]]:
        """(flux in V = 0 sources, valid, the intermediate quantities)"""
        radius = RADIUS_KM[name]
        d = float(np.linalg.norm(centre))
        separation = angle_deg(centre, b)
        rho = float(np.degrees(np.arcsin(radius / d)))
        limb = separation - rho
        azimuth = float(np.degrees(np.arctan2(centre @ x, centre @ y)))
        if name == "sun":
            phase = 0.0
            v_mag = self.c.sun_v_mag_1au + 5 * np.log10(d / AU_KM)
        else:
            phase = angle_deg(sun - centre, -centre)
            if name == "earth":
                full = self.c.sun_v_mag_1au - 2.5 * np.log10(
                    self.c.earth_albedo * (radius / AU_KM) ** 2
                )
                v_mag = (
                    full
                    + 5 * np.log10(d / AU_KM)
                    - 2.5 * np.log10(lambert_phase(radius / d, phase))
                )
            else:
                v_mag = (
                    self.c.moon_v_mag_full
                    + 5 * np.log10(d / 384400.0)
                    + 0.026 * phase
                    + 4e-9 * phase**4
                )
        info = {
            "v_mag": float(v_mag),
            "phase_deg": np.nan if name == "sun" else phase,
            "distance_km": d,
            "angular_radius_deg": rho,
            "separation_deg": separation,
            "limb_deg": limb,
            "azimuth_deg": azimuth,
            "pst_center": float(
                10.0 ** self.pst(np.array([separation]), np.array([azimuth]))[0]
            ),
            "pst_mean": np.nan,
            "lit_fraction": np.nan,
            "quadrature_error": np.nan,
            "status": float(re.StrayLightStatus.IN_FIELD),
        }
        in_field = limb < self.c.in_field_deg
        n_r, n_az = START_GRID[name]
        lambert = name != "sun"
        mean, lit, spread = self.level(b, y, x, centre, radius, sun, n_r, n_az, lambert)
        unsettled, change = True, np.inf
        for k in range(1, 5):
            finer = self.level(
                b, y, x, centre, radius, sun, n_r << k, n_az << k, lambert
            )
            change = abs(finer[0] - mean) / finer[0]
            mean, lit, spread = finer
            unsettled = change > 0.02 or lit == 0
            if not unsettled:
                break
        converged = spread <= 0.1 if lit == 0 else not unsettled
        dark = (
            name != "sun"
            and lit == 0
            and phase >= 90 + np.degrees(np.arccos(radius / d))
        )
        status = (
            re.StrayLightStatus.IN_FIELD
            if in_field
            else re.StrayLightStatus.DARK
            if dark
            else re.StrayLightStatus.FIRST_CELL
            if limb < self.c.pst.off_axis_deg[1]
            else re.StrayLightStatus.UNCONVERGED
            if not converged
            else re.StrayLightStatus.OK
        )
        info.update(
            pst_mean=mean,
            lit_fraction=lit,
            quadrature_error=spread if lit == 0 else change,
            status=float(status),
        )
        return float(10.0 ** (-0.4 * v_mag) * mean), status.is_valid, info

    def __call__(
        self,
        ra: float,
        dec: float,
        bodies: dict[str, Array],
        roll: float | None = None,
    ) -> dict[str, float]:
        r, d = np.radians(ra), np.radians(dec)
        b = np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
        sun = bodies["sun"]
        y, x, roll_used = self.frame(b, sun, roll)
        longitude, latitude = ecliptic(b)
        from_sun = (longitude - ecliptic(sun)[0]) % 360.0
        tabulated = zodi_table(b, sun)
        zodi = self.c.zodi_v_mag_arcsec2 if self.c.zodi == "constant" else tabulated
        per_zodi = self.c.conversion_factor / 10.0 ** (-0.4 * zodi)
        out: dict[str, float] = {"zodi_v_mag_arcsec2": zodi}
        out.update(
            roll_deg=roll_used,
            helio_ecliptic_longitude_deg=from_sun,
            ecliptic_latitude_deg=latitude,
            zodi_table_v_mag_arcsec2=tabulated,
        )
        total, all_valid = 0.0, True
        for name in ("sun", "earth", "moon"):
            flux, valid, info = self.body(name, b, y, x, bodies[name], sun)
            out[f"{name}_stray_light_ratio"] = flux * per_zodi if valid else np.nan
            total += flux
            all_valid &= valid
            out.update({f"{name}_{key}": value for key, value in info.items()})
        out["stray_light_ratio"] = total * per_zodi if all_valid else np.nan
        return out


def reference_values(
    constraint: StrayLightConstraint,
    ephem: Any,
    ra: Array,
    dec: Array,
    rows: Any,
    rolls: Any = None,
) -> dict[str, Array]:
    model = Reference(constraint)
    bodies = bodies_at(ephem, rows)
    cells = [
        model(
            float(ra[i]),
            float(dec[i]),
            {k: v[i] for k, v in bodies.items()},
            None if rolls is None else rolls[i],
        )
        for i in range(len(ra))
    ]
    return {key: np.array([cell[key] for cell in cells]) for key in cells[0]}


def assert_values_match(values: dict[str, Any], expected: dict[str, Array]) -> None:
    """Every reported value against the reference."""
    assert set(values) == set(expected)
    assert len(values) == 5 + 4 + 3 * 12
    for key, reference in expected.items():
        mine = np.asarray(values[key])
        if key.endswith("azimuth_deg") or key == "roll_deg":
            difference = (mine - reference + 180.0) % 360.0 - 180.0
            np.testing.assert_allclose(difference, 0.0, atol=1e-8, err_msg=key)
        elif key.endswith("quadrature_error"):
            # A change between levels of exactly zero is at rounding level.
            np.testing.assert_allclose(
                mine, reference, rtol=1e-6, atol=1e-11, equal_nan=True, err_msg=key
            )
        else:
            np.testing.assert_allclose(
                mine, reference, rtol=1e-9, equal_nan=True, err_msg=key
            )


# --- the model ---------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"sun_azimuth_deg": 180.0},
        {"sun_azimuth_deg": 180.0, "roll_offset_deg": 12.5},
        {"roll_offset_deg": -40.0, "zodi": "table"},
        {"earth_albedo": 0.4, "in_field_deg": 2.0, "zodi_v_mag_arcsec2": 21.9},
    ],
    ids=["default", "away_from_sun", "rolled", "zodi_table", "other_constants"],
)
def test_matches_independent_reference(
    table: PointSourceTransmittance,
    ephem: Any,
    sample: tuple[Array, Array, Any],
    options: dict[str, Any],
) -> None:
    ra, dec, rows = sample
    constraint = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, **options
    )
    result = evaluate(constraint, ephem, ra, dec, rows)
    expected = reference_values(constraint, ephem, ra, dec, rows)
    assert_values_match(result.constraint_values, expected)
    # Violated above the limit, and wherever there is no prediction.
    ratio = expected["stray_light_ratio"]
    np.testing.assert_array_equal(result.constraint_array, ~(ratio <= 3.0))
    # The sample exercises valid and invalid cells of every kind used below.
    assert np.isnan(ratio).any() and np.isfinite(ratio).sum() > len(ratio) // 2


@pytest.mark.parametrize("zodi", ["constant", "table"])
def test_per_body_values_explain_the_ratio(
    table: PointSourceTransmittance,
    ephem: Any,
    sample: tuple[Array, Array, Any],
    zodi: Literal["constant", "table"],
) -> None:
    """The intermediates are reported for every body, with or without a
    prediction, and are enough to rebuild the ratio."""
    ra, dec, rows = sample
    constraint = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, sun_azimuth_deg=180, zodi=zodi
    )
    result = evaluate(constraint, ephem, ra, dec, rows)
    values = {k: np.asarray(v) for k, v in result.constraint_values.items()}
    assert_values_match(values, reference_values(constraint, ephem, ra, dec, rows))

    # The statuses explain the validity, and the flux follows from the parts.
    statuses = {b: values[f"{b}_status"].astype(int) for b in ("sun", "earth", "moon")}
    seen = set(np.concatenate(list(statuses.values())).tolist())
    assert {int(re.StrayLightStatus.OK), int(re.StrayLightStatus.DARK)} <= seen
    assert seen <= {int(status) for status in re.StrayLightStatus}
    per_zodi = CONVERSION / 10.0 ** (-0.4 * values["zodi_v_mag_arcsec2"])
    for body, status in statuses.items():
        valid = np.isin(
            status, [int(re.StrayLightStatus.OK), int(re.StrayLightStatus.DARK)]
        )
        part = values[f"{body}_stray_light_ratio"]
        np.testing.assert_array_equal(np.isfinite(part), valid & np.isfinite(per_zodi))
        flux = 10.0 ** (-0.4 * values[f"{body}_v_mag"]) * values[f"{body}_pst_mean"]
        usable = np.isfinite(part)
        np.testing.assert_allclose(part[usable], (flux * per_zodi)[usable], rtol=1e-12)
    np.testing.assert_allclose(
        values["sun_limb_deg"],
        values["sun_separation_deg"] - values["sun_angular_radius_deg"],
        rtol=1e-12,
    )


def test_values_for_a_body_in_the_field(
    table: PointSourceTransmittance, ephem: Any
) -> None:
    row = 300
    moon = unit(bodies_at(ephem, [row])["moon"][0])
    ra = float(np.degrees(np.arctan2(moon[1], moon[0])) % 360)
    dec = float(np.degrees(np.arcsin(moon[2])))
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    values = evaluate(constraint, ephem, ra, dec, [row]).constraint_values
    assert values["moon_status"][0] == re.StrayLightStatus.IN_FIELD
    assert not re.StrayLightStatus(int(values["moon_status"][0])).is_valid
    assert values["moon_separation_deg"][0] < 1e-6
    assert values["moon_limb_deg"][0] < 0
    assert np.isfinite(values["moon_pst_mean"][0])
    assert np.isnan(values["moon_stray_light_ratio"][0])
    assert np.isnan(values["stray_light_ratio"][0])
    # In a tree the keys carry the constraint's prefix like any others.
    tree = constraint | re.SunConstraint(min_angle=45)
    assert (
        "stray_light.moon_status"
        in evaluate(tree, ephem, ra, dec, [row]).constraint_values
    )


def test_matches_reference_for_a_full_turn_table(
    ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows = (a[:60] for a in sample)
    constraint = StrayLightConstraint(
        pst=make_table(symmetric=False),
        conversion_factor=CONVERSION,
        sun_azimuth_deg=70,
    )
    values = evaluate(constraint, ephem, ra, dec, rows).constraint_values
    expected = reference_values(constraint, ephem, ra, dec, rows)
    np.testing.assert_allclose(
        np.asarray(values["stray_light_ratio"]),
        expected["stray_light_ratio"],
        rtol=1e-9,
        equal_nan=True,
    )
    # Without mirror symmetry the sign of the roll matters.
    mirrored = StrayLightConstraint(
        pst=make_table(symmetric=False),
        conversion_factor=CONVERSION,
        sun_azimuth_deg=-70,
    )
    other = evaluate(mirrored, ephem, ra, dec, rows).constraint_values
    assert not np.allclose(
        np.asarray(values["stray_light_ratio"]),
        np.asarray(other["stray_light_ratio"]),
        equal_nan=True,
    )


def test_symmetric_table_ignores_the_sign_of_the_sun_azimuth(
    table: PointSourceTransmittance, ephem: Any
) -> None:
    """Folding makes the Sun's own contribution even in its azimuth; the Earth
    and Moon sit elsewhere in the table, so theirs changes."""
    kwargs: dict[str, Any] = dict(pst=table, conversion_factor=CONVERSION)
    ra, dec, rows = np.array([200.0, 40.0]), np.array([30.0, -50.0]), [100, 700]
    plus = evaluate(
        StrayLightConstraint(sun_azimuth_deg=50, **kwargs), ephem, ra, dec, rows
    ).constraint_values
    minus = evaluate(
        StrayLightConstraint(sun_azimuth_deg=-50, **kwargs), ephem, ra, dec, rows
    ).constraint_values
    np.testing.assert_allclose(
        plus["sun_stray_light_ratio"], minus["sun_stray_light_ratio"], rtol=1e-12
    )
    assert not np.allclose(
        plus["earth_stray_light_ratio"], minus["earth_stray_light_ratio"]
    )


def test_tabulated_phase_law_matches_direct_integration() -> None:
    """The table the Earth's magnitude reads is the finite-distance Lambert
    integral: Phi(alpha) of a distant sphere at x = 0."""
    alpha = np.array([0.0, 45.0, 90.0, 135.0])
    a = np.radians(alpha)
    distant = (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi
    np.testing.assert_allclose(lambert_integral(0.0, alpha), distant, rtol=1e-5)
    np.testing.assert_allclose(
        [lambert_phase(0.0, x) for x in alpha], distant, rtol=1e-5
    )
    # Off the grid nodes the bilinear table stays close to the integral.
    for x, phase in [(0.137, 61.0), (0.211, 118.3), (0.05, 7.7)]:
        direct = lambert_integral(x, np.array([phase]))[0]
        assert lambert_phase(x, phase) == pytest.approx(direct, rel=5e-3)


def test_ratio_scales_with_the_conversion_factor_and_the_zodi(
    table: PointSourceTransmittance, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows = (a[:40] for a in sample)
    base = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    doubled = StrayLightConstraint(pst=table, conversion_factor=2 * CONVERSION)
    fainter = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, zodi_v_mag_arcsec2=23.5
    )
    reference = np.asarray(
        evaluate(base, ephem, ra, dec, rows).constraint_values["stray_light_ratio"]
    )
    for other, factor in ((doubled, 2.0), (fainter, 10**0.4)):
        ratio = np.asarray(
            evaluate(other, ephem, ra, dec, rows).constraint_values["stray_light_ratio"]
        )
        np.testing.assert_allclose(
            ratio, factor * reference, rtol=1e-12, equal_nan=True
        )


def test_total_is_the_sum_of_the_bodies(
    table: PointSourceTransmittance, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows = sample
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    values = {
        k: np.asarray(v)
        for k, v in evaluate(constraint, ephem, ra, dec, rows).constraint_values.items()
    }
    parts: Array = np.sum(
        [values[f"{b}_stray_light_ratio"] for b in ("sun", "earth", "moon")], axis=0
    )
    valid = np.isfinite(values["stray_light_ratio"])
    np.testing.assert_allclose(
        values["stray_light_ratio"][valid], parts[valid], rtol=1e-12
    )
    # The total is unknown exactly when a body is.
    np.testing.assert_array_equal(~valid, np.isnan(parts))


def test_body_in_the_field_is_a_violation_whatever_the_limit(
    table: PointSourceTransmittance, ephem: Any
) -> None:
    row = 300
    bodies = bodies_at(ephem, [row])
    constraint = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, max_ratio=1e300
    )
    for name in ("sun", "earth", "moon"):
        v = unit(bodies[name][0])
        ra = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
        dec = float(np.degrees(np.arcsin(v[2])))
        result = evaluate(constraint, ephem, ra, dec, [row])
        assert bool(result.constraint_array[0]), name
        values = result.constraint_values
        assert np.isnan(values["stray_light_ratio"][0])
        assert np.isnan(values[f"{name}_stray_light_ratio"][0])
    # Far from all three, the same limit is met.
    away = -unit(bodies["sun"][0] / AU_KM + unit(bodies["earth"][0]) * 0.2)
    ra = float(np.degrees(np.arctan2(away[1], away[0])) % 360)
    dec = float(np.degrees(np.arcsin(away[2])))
    result = evaluate(constraint, ephem, ra, dec, [row])
    assert not bool(result.constraint_array[0])
    assert np.isfinite(result.constraint_values["stray_light_ratio"][0])


def test_zodi_table_gap_is_a_violation(
    table: PointSourceTransmittance, ephem: Any
) -> None:
    """Toward the ecliptic 45 degrees from the Sun the table has no entry."""
    row = 500
    sun = unit(bodies_at(ephem, [row])["sun"][0])
    e = np.radians(23.4392911)
    pole = np.array([0.0, -np.sin(e), np.cos(e)])
    along = np.cross(pole, sun)
    b = np.cos(np.radians(45)) * sun + np.sin(np.radians(45)) * along
    ra, dec = (
        float(np.degrees(np.arctan2(b[1], b[0])) % 360),
        float(np.degrees(np.arcsin(b[2]))),
    )
    kwargs: dict[str, Any] = dict(
        pst=table, conversion_factor=CONVERSION, max_ratio=1e300
    )
    constant = evaluate(StrayLightConstraint(**kwargs), ephem, ra, dec, [row])
    tabulated = evaluate(
        StrayLightConstraint(zodi="table", **kwargs), ephem, ra, dec, [row]
    )
    assert not bool(constant.constraint_array[0])
    assert bool(tabulated.constraint_array[0])
    assert np.isnan(tabulated.constraint_values["zodi_v_mag_arcsec2"][0])
    assert np.isnan(tabulated.constraint_values["sun_stray_light_ratio"][0])


def test_single_column_table_does_not_depend_on_roll(ephem: Any) -> None:
    off_axis = np.arange(0.0, 151.0, 5.0)
    pst = PointSourceTransmittance.from_pst(
        off_axis, [0.0], 10.0 ** (-3.0 - 0.04 * off_axis)[:, None]
    )
    ra, dec, rows = np.array([10.0, 250.0]), np.array([20.0, -35.0]), [50, 900]
    ratios = [
        np.asarray(
            evaluate(
                StrayLightConstraint(
                    pst=pst, conversion_factor=CONVERSION, roll_offset_deg=roll
                ),
                ephem,
                ra,
                dec,
                rows,
            ).constraint_values["stray_light_ratio"]
        )
        for roll in (0.0, 77.0)
    ]
    assert np.isfinite(ratios[0]).all()
    np.testing.assert_allclose(ratios[0], ratios[1], rtol=1e-12)


# --- evaluation methods ------------------------------------------------------


@pytest.fixture(scope="module")
def constraint(table: PointSourceTransmittance) -> StrayLightConstraint:
    # A limit that splits the sample.
    return StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, max_ratio=40.0, sun_azimuth_deg=180
    )


def test_batch_grid_matches_elementwise(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, _ = (a[:12] for a in sample)
    rows = [0, 250, 600, 1200]
    times = [ephem.timestamp[i] for i in rows]
    grid = constraint.in_constraint_batch(ephem, list(ra), list(dec), times=times)
    assert grid.shape == (12, 4)
    assert grid.any() and not grid.all()
    for column, row in enumerate(rows):
        paired = evaluate(constraint, ephem, ra, dec, [row] * 12)
        np.testing.assert_array_equal(grid[:, column], paired.constraint_array)
    by_index = constraint.in_constraint_batch(ephem, list(ra), list(dec), indices=rows)
    np.testing.assert_array_equal(by_index, grid)
    paired = constraint.in_constraint_elementwise(ephem, ra[:4], dec[:4], times=times)
    np.testing.assert_array_equal(paired, np.diag(grid[:4]))


def test_evaluate_reports_windows_and_values(
    constraint: StrayLightConstraint, ephem: Any
) -> None:
    rows = list(range(0, N, 40))
    result = constraint.evaluate(ephem, 200.0, 35.0, indices=rows)
    paired = evaluate(constraint, ephem, [200.0] * len(rows), [35.0] * len(rows), rows)
    assert list(result.constraint_array) == list(paired.constraint_array)
    np.testing.assert_allclose(
        result.constraint_values["stray_light_ratio"],
        paired.constraint_values["stray_light_ratio"],
        rtol=1e-12,
        equal_nan=True,
    )
    assert result.all_satisfied == (not any(result.constraint_array))
    assert result.constraint_name == "StrayLight(max_ratio=40)"
    (batch,) = constraint.evaluate_batch(ephem, [200.0], [35.0], indices=rows)
    assert list(batch.constraint_array) == list(result.constraint_array)
    assert constraint.in_constraint(
        ephem.timestamp[rows[3]], ephem, 200.0, 35.0
    ) == bool(result.constraint_array[3])


def test_off_grid_times_are_interpolated(
    constraint: StrayLightConstraint, ephem: Any
) -> None:
    """Between samples the ratio lies between its neighbours (the geometry
    moves by well under a degree in ten minutes on this orbit)."""
    row = 800
    before, after = ephem.timestamp[row], ephem.timestamp[row + 1]
    middle = before + (after - before) / 2
    ratios = np.asarray(
        constraint.evaluate_elementwise(
            ephem, [200.0] * 3, [35.0] * 3, times=[before, middle, after]
        ).constraint_values["stray_light_ratio"]
    )
    assert np.isfinite(ratios).all()
    assert min(ratios[0], ratios[2]) <= ratios[1] <= max(ratios[0], ratios[2])
    assert ratios[0] != ratios[2]


def test_combinators_and_value_keys(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows = (a[:40] for a in sample)
    sun = re.SunConstraint(min_angle=60)
    alone = evaluate(constraint, ephem, ra, dec, rows)
    sun_alone = evaluate(sun, ephem, ra, dec, rows)
    either = evaluate(constraint | sun, ephem, ra, dec, rows)
    both = evaluate(constraint & sun, ephem, ra, dec, rows)
    np.testing.assert_array_equal(
        either.constraint_array,
        np.asarray(alone.constraint_array) | np.asarray(sun_alone.constraint_array),
    )
    np.testing.assert_array_equal(
        both.constraint_array,
        np.asarray(alone.constraint_array) & np.asarray(sun_alone.constraint_array),
    )
    assert "stray_light.stray_light_ratio" in either.constraint_values
    assert "sun.sun_angle_deg" in either.constraint_values
    np.testing.assert_allclose(
        either.constraint_values["stray_light.stray_light_ratio"],
        alone.constraint_values["stray_light_ratio"],
        rtol=1e-12,
        equal_nan=True,
    )
    inverted = evaluate(~constraint, ephem, ra, dec, rows)
    np.testing.assert_array_equal(
        inverted.constraint_array, ~np.asarray(alone.constraint_array)
    )
    windows = (constraint | sun).evaluate(ephem, 200.0, 35.0)
    causes = {c for w in windows.visibility for c in (w.start_cause or [])}
    assert causes <= {"stray_light", "sun"}


# --- explicit rolls ----------------------------------------------------------


def roll_inputs(
    ephem: Any, sample: tuple[Array, Array, Any], count: int = 24
) -> tuple[Array, Array, Any, list[Any]]:
    ra, dec, rows = (a[:count] for a in sample)
    return ra, dec, rows, [ephem.timestamp[int(i)] for i in rows]


def test_explicit_rolls_match_independent_reference(
    table: PointSourceTransmittance, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    """A target roll is the absolute spacecraft roll; roll_offset_deg does not
    apply to those rows."""
    ra, dec, rows, times = roll_inputs(ephem, sample)
    rolls = list(np.random.default_rng(3).uniform(-180.0, 360.0, len(ra)))
    constraint = StrayLightConstraint(
        pst=table,
        conversion_factor=CONVERSION,
        sun_azimuth_deg=180,
        roll_offset_deg=25.0,
    )
    result = constraint.evaluate_elementwise(
        ephem, ra, dec, times=times, target_rolls=rolls
    )
    expected = reference_values(constraint, ephem, ra, dec, rows, rolls)
    assert_values_match(result.constraint_values, expected)
    np.testing.assert_allclose(
        result.constraint_values["roll_deg"], np.mod(rolls, 360.0), atol=1e-9
    )
    np.testing.assert_array_equal(
        result.constraint_array,
        ~(expected["stray_light_ratio"] <= constraint.max_ratio),
    )


def test_reported_roll_reproduces_the_nominal_attitude(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    """roll_deg is the roll to pass back: the nominal value reproduces the
    result without a roll."""
    ra, dec, _, times = roll_inputs(ephem, sample)
    nominal = constraint.evaluate_elementwise(ephem, ra, dec, times=times)
    rolls = list(nominal.constraint_values["roll_deg"])
    again = constraint.evaluate_elementwise(
        ephem, ra, dec, times=times, target_rolls=rolls
    )
    np.testing.assert_array_equal(again.constraint_array, nominal.constraint_array)
    assert_values_match(
        again.constraint_values,
        {key: np.asarray(v) for key, v in nominal.constraint_values.items()},
    )


@pytest.mark.parametrize("offset", [12.5, -70.0])
def test_roll_plus_x_reproduces_roll_offset_x(
    table: PointSourceTransmittance,
    ephem: Any,
    sample: tuple[Array, Array, Any],
    offset: float,
) -> None:
    ra, dec, _, times = roll_inputs(ephem, sample)
    kwargs: dict[str, Any] = dict(
        pst=table, conversion_factor=CONVERSION, sun_azimuth_deg=180
    )
    nominal = StrayLightConstraint(**kwargs)
    rolled = StrayLightConstraint(roll_offset_deg=offset, **kwargs)
    base = np.asarray(
        nominal.evaluate_elementwise(ephem, ra, dec, times=times).constraint_values[
            "roll_deg"
        ]
    )
    by_offset = rolled.evaluate_elementwise(ephem, ra, dec, times=times)
    by_roll = nominal.evaluate_elementwise(
        ephem, ra, dec, times=times, target_rolls=list(base + offset)
    )
    np.testing.assert_allclose(
        by_roll.constraint_values["stray_light_ratio"],
        by_offset.constraint_values["stray_light_ratio"],
        rtol=1e-9,
        equal_nan=True,
    )
    difference = (
        np.asarray(by_offset.constraint_values["roll_deg"]) - base - offset + 180.0
    ) % 360.0 - 180.0
    np.testing.assert_allclose(difference, 0.0, atol=1e-9)


def test_mixed_rolls_in_one_call(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows, times = roll_inputs(ephem, sample, 12)
    rolls: list[float | None] = [None if i % 2 else 40.0 + 20.0 * i for i in range(12)]
    mixed = constraint.evaluate_elementwise(
        ephem, ra, dec, times=times, target_rolls=rolls
    )
    expected = reference_values(constraint, ephem, ra, dec, rows, rolls)
    assert_values_match(mixed.constraint_values, expected)
    mask = constraint.in_constraint_elementwise(
        ephem, ra, dec, times=times, target_rolls=rolls
    )
    np.testing.assert_array_equal(mask, mixed.constraint_array)


ROLLS = np.arange(0.0, 360.0, 5.0)


@pytest.fixture(scope="module")
def roll_case(
    table: PointSourceTransmittance, ephem: Any, sample: tuple[Array, Array, Any]
) -> tuple[StrayLightConstraint, float, float, Any, Array]:
    """A pointing with a prediction at every roll, and a constraint whose limit
    splits the rolls: (constraint, ra, dec, time, ratio at each of ROLLS)."""
    probe = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    n = len(ROLLS)
    for ra, dec, row in zip(*sample):
        time = ephem.timestamp[int(row)]
        ratios = np.asarray(
            probe.evaluate_elementwise(
                ephem, [ra] * n, [dec] * n, times=[time] * n, target_rolls=list(ROLLS)
            ).constraint_values["stray_light_ratio"]
        )
        if np.isfinite(ratios).all() and ratios.max() > 1.5 * ratios.min():
            limit = float(np.sqrt(ratios.max() * ratios.min()))
            constraint = StrayLightConstraint(
                pst=table, conversion_factor=CONVERSION, max_ratio=limit
            )
            return constraint, float(ra), float(dec), time, ratios
    raise AssertionError("no pointing in the sample varies with roll")


def test_every_method_takes_the_roll(
    roll_case: tuple[StrayLightConstraint, float, float, Any, Array], ephem: Any
) -> None:
    """The same roll gives the same answer through every evaluation method."""
    constraint, ra, dec, time, ratios = roll_case
    nominal = bool(constraint.in_constraint(time, ephem, ra, dec))
    for index in (int(np.argmax(ratios)), int(np.argmin(ratios))):
        roll, ratio = float(ROLLS[index]), ratios[index]
        expected = bool(ratio > constraint.max_ratio)
        single = constraint.evaluate(ephem, ra, dec, times=[time], target_roll=roll)
        assert bool(single.constraint_array[0]) == expected
        assert single.constraint_values["stray_light_ratio"][0] == pytest.approx(ratio)
        assert single.constraint_values["roll_deg"][0] == pytest.approx(roll)
        (batch,) = constraint.evaluate_batch(
            ephem, [ra], [dec], times=[time], target_rolls=[roll]
        )
        assert batch.constraint_values["stray_light_ratio"][0] == pytest.approx(ratio)
        grid = constraint.in_constraint_batch(
            ephem, [ra, ra], [dec, dec], times=[time], target_rolls=[roll, None]
        )
        assert [bool(grid[0, 0]), bool(grid[1, 0])] == [expected, nominal]
        assert (
            bool(constraint.in_constraint(time, ephem, ra, dec, target_roll=roll))
            == expected
        )


def test_combined_constraint_shares_the_roll(
    table: PointSourceTransmittance, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    """In a tree, the stray-light leaf and a solar-roll leaf see the same
    spacecraft roll, and the nominal attitude is the solar-optimal roll."""
    ra, dec, _, times = roll_inputs(ephem, sample, 16)
    stray = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, max_ratio=40.0
    )
    solar = re.SolarRollConstraint(tolerance_deg=5.0)
    nominal = stray.evaluate_elementwise(ephem, ra, dec, times=times)
    rolls = np.asarray(nominal.constraint_values["roll_deg"])
    optimal = np.asarray(
        solar.evaluate_elementwise(
            ephem, ra, dec, times=times, target_rolls=list(rolls)
        ).constraint_values["solar_optimal_roll_deg"]
    )
    np.testing.assert_allclose(
        (rolls - optimal + 180.0) % 360.0 - 180.0, 0.0, atol=1e-9
    )

    tree = stray | solar
    for shift in (0.0, 30.0):
        given = list(rolls + shift)
        combined = tree.evaluate_elementwise(
            ephem, ra, dec, times=times, target_rolls=given
        )
        alone = stray.evaluate_elementwise(
            ephem, ra, dec, times=times, target_rolls=given
        )
        np.testing.assert_allclose(
            combined.constraint_values["stray_light.stray_light_ratio"],
            alone.constraint_values["stray_light_ratio"],
            rtol=1e-12,
            equal_nan=True,
        )
        # 30 degrees off the solar-optimal roll breaks the 5 degree tolerance.
        solar_violated = shift > 5.0
        np.testing.assert_array_equal(
            combined.constraint_array,
            np.asarray(alone.constraint_array) | solar_violated,
        )


def test_per_row_rolls_in_a_tree_of_roll_independent_rules(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    """With rules that ignore the roll, a tree is evaluated in one paired pass:
    the result is each rule's own, combined."""
    ra, dec, _, times = roll_inputs(ephem, sample, 40)
    rolls: list[float | None] = [None if i % 3 == 0 else 17.0 * i for i in range(40)]
    sun = re.SunConstraint(min_angle=60)
    limb = re.EarthLimbConstraint(min_angle=15)
    kwargs: dict[str, Any] = dict(times=times, target_rolls=rolls)
    stray_alone = constraint.evaluate_elementwise(ephem, ra, dec, **kwargs)
    # The roll means nothing to these two.
    sun_alone = np.asarray(sun.in_constraint_elementwise(ephem, ra, dec, times=times))
    limb_alone = np.asarray(limb.in_constraint_elementwise(ephem, ra, dec, times=times))
    np.testing.assert_array_equal(
        sun.in_constraint_elementwise(ephem, ra, dec, **kwargs), sun_alone
    )
    stray = np.asarray(stray_alone.constraint_array)
    cases = {
        "or": (constraint | sun | limb, stray | sun_alone | limb_alone),
        "and_not": (constraint & ~sun, stray & ~sun_alone),
        "xor": (constraint ^ limb, stray ^ limb_alone),
    }
    for name, (tree, expected) in cases.items():
        result = tree.evaluate_elementwise(ephem, ra, dec, **kwargs)
        np.testing.assert_array_equal(result.constraint_array, expected, err_msg=name)
        np.testing.assert_array_equal(
            tree.in_constraint_elementwise(ephem, ra, dec, **kwargs),
            expected,
            err_msg=name,
        )
        for key, values in stray_alone.constraint_values.items():
            np.testing.assert_array_equal(
                result.constraint_values[f"stray_light.{key}"],
                values,
                err_msg=f"{name} {key}",
            )
    assert stray.any() and not stray.all()


def test_many_distinct_rolls_cost_about_the_same_as_none(
    constraint: StrayLightConstraint, ephem: Any
) -> None:
    """Rows are independent, so distinct rolls and times need no grouping."""
    import time as clock

    rng = np.random.default_rng(9)
    count = 600
    ra = rng.uniform(0.0, 360.0, count)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count)))
    times = [ephem.timestamp[int(i)] for i in rng.integers(0, N, count)]
    rolls = list(rng.uniform(0.0, 360.0, count))

    def best(**kwargs: Any) -> float:
        elapsed = []
        for _ in range(3):
            start = clock.perf_counter()
            constraint.evaluate_elementwise(ephem, ra, dec, times=times, **kwargs)
            elapsed.append(clock.perf_counter() - start)
        return min(elapsed)

    # Grouped by (time, roll) this took tens of times longer than no roll.
    assert best(target_rolls=rolls) < 5.0 * best() + 0.05


def test_roll_range_gives_the_rolls_that_are_dark(
    roll_case: tuple[StrayLightConstraint, float, float, Any, Array], ephem: Any
) -> None:
    constraint, ra, dec, time, ratios = roll_case
    violated = ratios > constraint.max_ratio
    assert violated.any() and not violated.all()
    ranges = constraint.roll_range(time, ephem, ra, dec, n_roll_samples=len(ROLLS))
    allowed = np.zeros(len(ROLLS), dtype=bool)
    for low, high in ranges:
        allowed |= (ROLLS >= low - 1e-9) & (ROLLS <= high + 1e-9)
    np.testing.assert_array_equal(allowed, ~violated)


def test_apparent_angles_shift_the_result_slightly(
    constraint: StrayLightConstraint, ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    ra, dec, rows = (a[:40] for a in sample)
    times = [ephem.timestamp[int(i)] for i in rows]
    geometric = np.asarray(
        constraint.evaluate_elementwise(ephem, ra, dec, times=times).constraint_values[
            "stray_light_ratio"
        ]
    )
    apparent = np.asarray(
        constraint.evaluate_elementwise(
            ephem, ra, dec, times=times, apparent=True
        ).constraint_values["stray_light_ratio"]
    )
    valid = np.isfinite(geometric) & np.isfinite(apparent)
    assert valid.sum() > 20
    # Tens of arcseconds across a table that varies by 12% per degree.
    np.testing.assert_allclose(apparent[valid], geometric[valid], rtol=5e-3)
    assert not np.array_equal(apparent[valid], geometric[valid])


# --- configuration -----------------------------------------------------------


class TestPointSourceTransmittance:
    def test_accepts_arrays_and_marks_empty_cells(self) -> None:
        log = np.array([[-1.0, np.nan], [-2.0, -2.5], [-3.0, -3.5]])
        pst = PointSourceTransmittance.model_validate(
            {
                "off_axis_deg": np.array([0.0, 10.0, 20.0]),
                "azimuth_deg": np.array([0.0, 180.0]),
                "log10_pst": log,
            }
        )
        assert pst.off_axis_deg == [0.0, 10.0, 20.0]
        assert pst.log10_pst[0] == [-1.0, None]
        assert pst.azimuth_symmetric is True

    def test_from_pst_takes_linear_values(self) -> None:
        pst = PointSourceTransmittance.from_pst(
            [0.0, 10.0], [0.0, 90.0], [[1e-2, 1e-3], [0.0, 1e-5]]
        )
        assert pst.log10_pst[0] == pytest.approx([-2.0, -3.0])
        assert pst.log10_pst[1][0] is None
        assert pst.log10_pst[1][1] == pytest.approx(-5.0)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(off_axis_deg=[0.0]), "at least two"),
            (dict(off_axis_deg=[0.0, 10.0, 5.0]), "strictly increasing"),
            (dict(off_axis_deg=[-5.0, 10.0, 20.0]), r"\[0, 180\]"),
            (dict(azimuth_deg=[0.0, 200.0]), "symmetric table"),
            (dict(azimuth_deg=[90.0, 0.0]), "strictly increasing"),
            (dict(azimuth_deg=[0.0, 400.0], azimuth_symmetric=False), "less than 360"),
            (dict(log10_pst=[[-1.0, -2.0], [-2.0, -3.0]]), "shape"),
            (dict(log10_pst=[[None, None]] * 3), "no finite value"),
        ],
    )
    def test_rejects_malformed_tables(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        base: dict[str, Any] = dict(
            off_axis_deg=[0.0, 10.0, 20.0],
            azimuth_deg=[0.0, 180.0],
            log10_pst=[[-1.0, -2.0], [-2.0, -3.0], [-3.0, -4.0]],
        )
        with pytest.raises(ValidationError, match=message):
            PointSourceTransmittance(**{**base, **kwargs})


class TestStrayLightConstraintModel:
    def test_defaults(self, table: PointSourceTransmittance) -> None:
        c = StrayLightConstraint(pst=table, conversion_factor=1e-9)
        assert c.type == "stray_light"
        assert c.max_ratio == 3.0
        assert c.roll_offset_deg == 0.0
        assert c.zodi == "constant"
        assert c.zodi_v_mag_arcsec2 == 22.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(conversion_factor=0.0),
            dict(conversion_factor=float("nan")),
            dict(max_ratio=-1.0),
            dict(in_field_deg=-0.1),
            dict(earth_albedo=0.0),
            dict(zodi="sky"),
            dict(roll_offset_deg=float("inf")),
        ],
    )
    def test_rejects_bad_values(
        self, table: PointSourceTransmittance, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            StrayLightConstraint(**{"pst": table, "conversion_factor": 1e-9, **kwargs})

    def test_json_round_trip(self, table: PointSourceTransmittance) -> None:
        c = StrayLightConstraint(
            pst=table, conversion_factor=1e-9, zodi="table", roll_offset_deg=5
        )
        assert StrayLightConstraint.model_validate_json(c.model_dump_json()) == c
        tree = c | re.SunConstraint(min_angle=45)
        restored = type(tree).model_validate_json(tree.model_dump_json())
        assert restored == tree
        assert isinstance(restored.constraints[0], StrayLightConstraint)

    def test_toml_round_trip(self, table: PointSourceTransmittance) -> None:
        c = StrayLightConstraint(pst=table, conversion_factor=1e-9, max_ratio=5.0)
        text = re.constraint_to_toml_string(c)
        assert "[pst]" in text
        assert re.parse_constraint_toml(text) == c
        tree = c | re.EarthLimbConstraint(min_angle=10)
        assert re.parse_constraint_toml(re.constraint_to_toml_string(tree)) == tree

    def test_rust_rejects_a_malformed_table(self) -> None:
        import json

        config: dict[str, Any] = {
            "type": "stray_light",
            "conversion_factor": 1e-9,
            "pst": {
                "off_axis_deg": [0.0, 10.0],
                "azimuth_deg": [0.0],
                "log10_pst": [[-1.0], [-2.0], [-3.0]],
            },
        }
        with pytest.raises(ValueError, match="Invalid stray_light constraint.*shape"):
            re.Constraint.from_json(json.dumps(config))
        config["pst"]["log10_pst"] = [[-1.0], [-2.0]]
        # The remaining fields take their defaults.
        assert "StrayLight" in repr(re.Constraint.from_json(json.dumps(config)))
