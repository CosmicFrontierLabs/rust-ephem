"""StrayLightConstraint against an independent NumPy implementation of the
model, on a synthetic point-source transmittance table."""

from datetime import datetime, timedelta, timezone
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


def along(p: Array, v: Array) -> Array:
    """Each row of p dotted with v (without BLAS, whose matmul raises spurious
    floating-point warnings on some platforms)."""
    return np.asarray(np.einsum("ij,j->i", p, v))


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


def lit_half_width(
    cos_alpha: float, sin_alpha: float, theta_max: float
) -> float | None:
    """Half the range of longitudes about the sub-observer point that hold
    sunlit surface inside the visible cap; None when there is none."""
    if cos_alpha > 0.0:
        return float(np.pi)
    edge = -cos_alpha * np.cos(theta_max)
    reach = sin_alpha * np.sin(theta_max)
    if not reach > edge:
        return None
    return float(np.arccos(np.clip(edge / reach, -1.0, 1.0)))


class Level:
    """One quadrature level: sum of weight x PST over the samples in view, the
    sum of the weights, and the solid angle of the sunlit samples in view."""

    def __init__(self, weighted: float, weight: float, lit: float) -> None:
        self.weighted, self.weight, self.lit = weighted, weight, lit

    @property
    def mean(self) -> float:
        return self.weighted / self.weight if self.weight > 0 else 0.0


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

    def pst_toward(self, b: Array, y: Array, x: Array, p: Array) -> Array:
        polar = np.degrees(np.arccos(np.clip(along(p, b), -1, 1)))
        azimuth = np.degrees(np.arctan2(along(p, x), along(p, y)))
        return np.asarray(10.0 ** self.pst(polar, azimuth))

    @staticmethod
    def hides(earth: Array | None, p: Array, reach: Any) -> Array:
        """Whether the Earth is on the line of sight along p before `reach`."""
        if earth is None:
            return np.zeros(len(p), dtype=bool)
        ahead = along(p, earth)
        discriminant = ahead**2 - (earth @ earth - RADIUS_KM["earth"] ** 2)
        root = np.sqrt(np.clip(discriminant, 0, None))
        return np.asarray((ahead > 0) & (discriminant > 0) & (ahead - root < reach))

    def sun_level(
        self,
        b: Array,
        y: Array,
        x: Array,
        centre: Array,
        n_r: int,
        n_az: int,
        earth: Array | None,
    ) -> Level:
        """The Sun's uniform disk, in angle from its centre and position angle."""
        d = float(np.linalg.norm(centre))
        e = centre / d
        rho = np.arcsin(RADIUS_KM["sun"] / d)
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
        weight = np.sin(gamma) * (rho / n_r) * (2 * np.pi / n_az)
        seen = ~self.hides(earth, p, np.inf)
        value = self.pst_toward(b, y, x, p)
        return Level(
            float(np.sum((weight * value)[seen])),
            float(weight.sum()),
            float(weight[seen].sum()),
        )

    def cap_level(
        self,
        b: Array,
        y: Array,
        x: Array,
        centre: Array,
        radius: float,
        to_sun: Array,
        n_theta: int,
        n_phi: int,
        earth: Array | None,
    ) -> Level:
        """The sunlit part of the visible cap of a Lambert sphere, in colatitude
        from the sub-observer point and longitude about it (zero on the Sun's
        side). Each longitude is sampled between the terminator and the edge
        of the cap, and only the longitudes holding sunlit surface."""
        d = float(np.linalg.norm(centre))
        toward = -centre / d
        theta_max = float(np.arccos(radius / d))
        cos_alpha = float(to_sun @ toward)
        across = to_sun - cos_alpha * toward
        sin_alpha = float(np.linalg.norm(across))
        phi_max = lit_half_width(cos_alpha, sin_alpha, theta_max)
        if phi_max is None:
            return Level(0.0, 0.0, 0.0)
        if sin_alpha > 1e-12:
            side = across / sin_alpha
        else:
            ref = (
                np.array([0.0, 0.0, 1.0])
                if abs(toward[2]) < 0.95
                else np.array([0.0, 1.0, 0.0])
            )
            side = unit(np.cross(ref, toward))
        third = np.cross(toward, side)
        d_phi = 2 * phi_max / n_phi
        phi: Array = -phi_max + (np.arange(n_phi) + 0.5) * d_phi
        toward_sun: Array = sin_alpha * np.cos(phi)
        low: Array
        high: Array
        with np.errstate(divide="ignore", invalid="ignore"):
            if cos_alpha > 0:
                low = np.zeros(n_phi)
                high = np.where(
                    toward_sun < 0,
                    np.minimum(theta_max, np.arctan(cos_alpha / -toward_sun)),
                    theta_max,
                )
            else:
                low = np.arctan(-cos_alpha / toward_sun)
                high = np.full(n_phi, theta_max)
        used = high > low
        phi, toward_sun, low, high = phi[used], toward_sun[used], low[used], high[used]
        d_theta = (high - low) / n_theta
        theta = low[:, None] + (np.arange(n_theta) + 0.5)[None, :] * d_theta[:, None]
        outward = np.cos(phi)[:, None] * side + np.sin(phi)[:, None] * third
        normal = (
            np.cos(theta)[..., None] * toward
            + np.sin(theta)[..., None] * outward[:, None, :]
        )
        point = centre + radius * normal
        reach = np.linalg.norm(point, axis=-1)
        cos_emission = (d * np.cos(theta) - radius) / reach
        solid = (
            cos_emission * radius**2 * np.sin(theta) * d_theta[:, None] * d_phi
        ) / reach**2
        weight = (
            np.clip(
                cos_alpha * np.cos(theta) + toward_sun[:, None] * np.sin(theta), 0, None
            )
            * solid
        )
        p = (point / reach[..., None]).reshape(-1, 3)
        seen = ~self.hides(earth, p, reach.ravel())
        value = self.pst_toward(b, y, x, p)
        return Level(
            float(np.sum((weight.ravel() * value)[seen])),
            float(weight.sum()),
            float(solid.ravel()[seen].sum()),
        )

    @staticmethod
    def hidden(name: str, centre: Array, earth: Array) -> str:
        """How much of a body the Earth hides: "none", "partly" or "all"."""
        if name == "earth":
            return "none"
        d, d_earth = float(np.linalg.norm(centre)), float(np.linalg.norm(earth))
        radius, earth_radius = RADIUS_KM[name], RADIUS_KM["earth"]
        separation = np.radians(angle_deg(centre, earth))
        rho_earth, rho = np.arcsin(earth_radius / d_earth), np.arcsin(radius / d)
        tangent = np.sqrt(d_earth**2 - earth_radius**2)
        if separation >= rho_earth + rho or d + radius <= tangent:
            return "none"
        if separation + rho <= rho_earth and d - radius >= tangent:
            return "all"
        return "partly"

    def body(
        self,
        name: str,
        b: Array,
        y: Array,
        x: Array,
        centre: Array,
        sun: Array,
        earth: Array,
    ) -> tuple[float, bool, dict[str, float]]:
        """(flux in V = 0 sources, valid, the intermediate quantities)"""
        status_of = re.StrayLightStatus
        radius = RADIUS_KM[name]
        d = float(np.linalg.norm(centre))
        separation = angle_deg(centre, b)
        rho = float(np.degrees(np.arcsin(radius / d)))
        limb = separation - rho
        azimuth = float(np.degrees(np.arctan2(centre @ x, centre @ y)))
        sunward = sun - centre
        sun_au = float(np.linalg.norm(sunward)) / AU_KM
        info = {
            "v_mag": np.nan,
            "phase_deg": np.nan,
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
            "status": float(status_of.IN_FIELD),
        }
        dark = False
        if name == "sun":
            info["v_mag"] = self.c.sun_v_mag_1au + 5 * np.log10(d / AU_KM)
        else:
            phase = angle_deg(sunward, -centre)
            info["phase_deg"] = phase
            a = np.radians(phase)
            dark = (
                lit_half_width(
                    float(np.cos(a)), float(np.sin(a)), float(np.arccos(radius / d))
                )
                is None
            )
            if name == "moon":
                info["v_mag"] = (
                    self.c.moon_v_mag_full
                    + 5 * np.log10(d / 384400.0)
                    + 5 * np.log10(sun_au)
                    + 0.026 * phase
                    + 4e-9 * phase**4
                )
        nothing = {"pst_mean": 0.0, "lit_fraction": 0.0}
        hidden = self.hidden(name, centre, earth)
        if hidden == "all":
            info.update(nothing, status=float(status_of.OCCULTED))
            return 0.0, True, info
        in_field = limb < self.c.in_field_deg
        if dark:
            info.update(nothing)
            if name == "earth":
                info["v_mag"] = np.inf
            if in_field:
                return np.nan, False, info
            info["status"] = float(status_of.DARK)
            return 0.0, True, info

        blocker = earth if hidden == "partly" else None
        n_a, n_b = START_GRID[name]

        def level(k: int) -> Level:
            if name == "sun":
                return self.sun_level(b, y, x, centre, n_a << k, n_b << k, blocker)
            return self.cap_level(
                b, y, x, centre, radius, unit(sunward), n_a << k, n_b << k, blocker
            )

        def signal(lv: Level) -> float:
            return lv.weighted if name == "earth" else lv.mean

        current = level(0)
        change = np.inf
        for k in range(1, 5):
            finer = level(k)
            new, old = signal(finer), signal(current)
            change = 0.0 if new == old else abs(new - old) / abs(new)
            current = finer
            if change <= 0.02:
                break
        converged = change <= 0.02
        if name == "earth":
            sun_at_earth = self.c.sun_v_mag_1au + 5 * np.log10(sun_au)
            radiance = (
                3 / (2 * np.pi) * self.c.earth_albedo * 10.0 ** (-0.4 * sun_at_earth)
            )
            info["v_mag"] = float(-2.5 * np.log10(radiance * current.weight))
        disk = 2 * np.pi * (1 - np.sqrt(1 - (radius / d) ** 2))
        status = (
            status_of.IN_FIELD
            if in_field
            else status_of.OCCULTED
            if blocker is not None and current.lit == 0
            else status_of.FIRST_CELL
            if limb < self.c.pst.off_axis_deg[1]
            else status_of.UNCONVERGED
            if not converged
            else status_of.OK
        )
        info.update(
            pst_mean=current.mean,
            lit_fraction=min(current.lit / disk, 1.0),
            quadrature_error=change,
            status=float(status),
        )
        flux = float(10.0 ** (-0.4 * info["v_mag"]) * current.mean)
        return flux, status.is_valid, info

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
            flux, valid, info = self.body(
                name, b, y, x, bodies[name], sun, bodies["earth"]
            )
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
    kwargs: dict[str, Any] = {"pst": table, "conversion_factor": CONVERSION}
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
    kwargs: dict[str, Any] = {
        "pst": table,
        "conversion_factor": CONVERSION,
        "max_ratio": 1e300,
    }
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


# --- low orbit: the Earth fills the sky, and hides things ---------------------

LOW_STEP = 2
LOW_N = 2850  # one revolution at 500 km


@pytest.fixture(scope="module")
def low_orbit(ensure_planetary_data: None) -> Any:
    """A circular orbit 500 km up, from which the Earth is 136 degrees across."""
    return circular_orbit(500.0, LOW_N, LOW_STEP)


def circular_orbit(altitude_km: float, count: int, step: int) -> Any:
    radius = RADIUS_KM["earth"] + altitude_km
    rate = np.sqrt(398600.4418 / radius**3)
    angle = rate * np.arange(count) * float(step)
    inclination = np.radians(30.0)
    along = np.array([1.0, 0.0, 0.0])
    across = np.array([0.0, np.cos(inclination), np.sin(inclination)])
    position = radius * (
        np.cos(angle)[:, None] * along + np.sin(angle)[:, None] * across
    )
    velocity = (
        radius
        * rate
        * (-np.sin(angle)[:, None] * along + np.cos(angle)[:, None] * across)
    )
    times = [BEGIN + timedelta(seconds=float(step * i)) for i in range(count)]
    return re.FileEphemeris.from_arrays(times, position, velocity, step_size=step)


def radec_of(v: Array) -> tuple[float, float]:
    v = unit(np.asarray(v, dtype=float))
    return (
        float(np.degrees(np.arctan2(v[1], v[0])) % 360.0),
        float(np.degrees(np.arcsin(v[2]))),
    )


def earth_phase_deg(ephem: Any, rows: Any) -> Array:
    """Sun-Earth-observer angle at each row."""
    bodies = bodies_at(ephem, rows)
    to_sun, to_observer = unit(bodies["sun"] - bodies["earth"]), unit(-bodies["earth"])
    return np.asarray(np.degrees(np.arccos(np.sum(to_sun * to_observer, axis=1))))


def sky_integral(
    constraint: StrayLightConstraint,
    b: Array,
    earth: Array,
    sun: Array,
    count: int = 500,
) -> float:
    """The Earth's scattered flux (V = 0 sources) summed over sky directions:
    radiance (3 p / 2 pi) E_sun cos(i) times PST times solid angle.

    Independent of the model's surface integral: a grid in angle from the
    Earth's centre and position angle, packed toward the limb, where a
    crescent lies, by gamma = rho (1 - s^2)."""
    model = Reference(constraint)
    y, x, _ = model.frame(b, sun, None)
    d = float(np.linalg.norm(earth))
    e = earth / d
    rho = np.arcsin(RADIUS_KM["earth"] / d)
    u = unit(np.cross([0.0, 0.0, 1.0], e))
    w = np.cross(e, u)
    s = (np.arange(count) + 0.5) / count
    gamma = rho * (1 - s**2)
    d_gamma = rho * 2 * s / count
    beta = (np.arange(2 * count) + 0.5) / (2 * count) * 2 * np.pi
    to_sun = unit(sun - earth)
    sun_at_earth = constraint.sun_v_mag_1au + 5 * np.log10(
        np.linalg.norm(sun - earth) / AU_KM
    )
    radiance = 3 / (2 * np.pi) * constraint.earth_albedo * 10.0 ** (-0.4 * sun_at_earth)
    total = 0.0
    for g, dg in zip(gamma, d_gamma):
        p = np.cos(g) * e + np.sin(g) * (
            np.cos(beta)[:, None] * u + np.sin(beta)[:, None] * w
        )
        reach = d * np.cos(g)
        hit = reach - np.sqrt(max(reach**2 - (d * d - RADIUS_KM["earth"] ** 2), 0.0))
        normal = unit(hit * p - earth)
        cos_incidence = np.clip(along(normal, to_sun), 0, None)
        value = model.pst_toward(b, y, x, p)
        total += float(np.sum(cos_incidence * value)) * np.sin(g) * dg
    return float(radiance * total * (2 * np.pi / (2 * count)))


def clear_of_the_earth(earth: Array, sun: Array, above_limb_deg: float) -> Array:
    """A boresight `above_limb_deg` beyond the Earth's limb, on the Sun's side
    of the nadir when there is one."""
    nadir = unit(earth)
    s = unit(sun)
    across = s - (s @ nadir) * nadir
    if np.linalg.norm(across) < 1e-6:
        across = np.cross(nadir, [0.0, 0.0, 1.0])
    across = unit(across)
    angle = np.arcsin(RADIUS_KM["earth"] / np.linalg.norm(earth)) + np.radians(
        above_limb_deg
    )
    return np.asarray(np.cos(angle) * nadir + np.sin(angle) * across)


@pytest.mark.parametrize("flat", [True, False], ids=["constant_pst", "smooth_pst"])
@pytest.mark.parametrize("altitude_km", [400.0, 500.0, 1000.0, 6000.0, 35786.0])
def test_earth_flux_is_radiance_times_pst_summed_over_the_sky(
    table: PointSourceTransmittance,
    ensure_planetary_data: None,
    altitude_km: float,
    flat: bool,
) -> None:
    """Against an independent integral over sky directions, from low orbit to
    geostationary height and from full Earth to a crescent. A factor for the
    inclination of each ray to the Earth's direction would lose a third of
    the light at 500 km (and three quarters of a crescent's)."""
    period = 2 * np.pi * np.sqrt((RADIUS_KM["earth"] + altitude_km) ** 3 / 398600.4418)
    count = 48
    ephem = circular_orbit(altitude_km, count, max(1, int(period / count)))
    off_axis = np.arange(0.0, 181.0, 5.0)
    pst = (
        PointSourceTransmittance.from_pst(
            off_axis, [0.0, 180.0], np.full((37, 2), 1e-6)
        )
        if flat
        else table
    )
    constraint = StrayLightConstraint(pst=pst, conversion_factor=CONVERSION)
    per_zodi = CONVERSION / 10.0 ** (-0.4 * constraint.zodi_v_mag_arcsec2)
    rows = np.arange(count)
    phase = earth_phase_deg(ephem, rows)
    reach = np.degrees(
        np.arccos(RADIUS_KM["earth"] / (RADIUS_KM["earth"] + altitude_km))
    )
    bodies = bodies_at(ephem, rows)
    checked = []
    # A dozen rows from full Earth to the thinnest crescent.
    lit = rows[phase < 90.0 + reach - 2.0]
    for row in lit[np.argsort(phase[lit])][:: max(1, len(lit) // 12)]:
        earth, sun = bodies["earth"][row], bodies["sun"][row]
        b = clear_of_the_earth(earth, sun, 25.0)
        ra, dec = radec_of(b)
        values = evaluate(constraint, ephem, ra, dec, [row]).constraint_values
        if values["earth_status"][0] != re.StrayLightStatus.OK:
            continue
        flux = values["earth_stray_light_ratio"][0] / per_zodi
        expected = sky_integral(constraint, b, earth, sun)
        # The quadrature stops when two levels agree to 2%.
        assert flux == pytest.approx(expected, rel=0.01 if flat else 0.03), (
            row,
            phase[row],
        )
        checked.append(phase[row])
    # Full Earth and crescent both.
    assert min(checked) < 30.0 and max(checked) > 90.0


def test_low_orbit_matches_independent_reference(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    """Every reported value, around a whole orbit: day, terminator, eclipse."""
    rng = np.random.default_rng(11)
    count = 240
    ra = rng.uniform(0.0, 360.0, count)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count)))
    rows = rng.integers(0, LOW_N, count)
    constraint = StrayLightConstraint(
        pst=table, conversion_factor=CONVERSION, sun_azimuth_deg=180
    )
    result = evaluate(constraint, low_orbit, ra, dec, rows)
    expected = reference_values(constraint, low_orbit, ra, dec, rows)
    assert_values_match(result.constraint_values, expected)
    seen = {
        int(v) for body in ("sun", "earth", "moon") for v in expected[f"{body}_status"]
    }
    status = re.StrayLightStatus
    assert {status.OK, status.DARK, status.OCCULTED, status.IN_FIELD} <= seen


def test_no_cell_is_unconverged_through_the_terminator(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    """Where the Earth is a crescent on the limb, the lit surface is still
    integrated: nothing is rejected for want of a prediction."""
    rng = np.random.default_rng(5)
    count = 60
    ra = rng.uniform(0.0, 360.0, count)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count)))
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    rows = np.arange(0, LOW_N, 15)
    phase = earth_phase_deg(low_orbit, rows)
    reach = np.degrees(np.arccos(RADIUS_KM["earth"] / (RADIUS_KM["earth"] + 500.0)))
    crescent = (phase > 95.0) & (phase < 90.0 + reach)
    assert crescent.sum() >= 8
    status = re.StrayLightStatus
    for row, alpha in zip(rows, phase):
        values = evaluate(
            constraint, low_orbit, ra, dec, [int(row)] * count
        ).constraint_values
        earth = np.asarray(values["earth_status"]).astype(int)
        assert status.UNCONVERGED not in earth, (row, alpha)
        in_field = earth == status.IN_FIELD
        # Dark exactly when the day side and the visible cap do not meet.
        dark = alpha >= 90.0 + reach
        np.testing.assert_array_equal(earth[~in_field] == status.DARK, dark)
        if dark:
            np.testing.assert_array_equal(
                np.asarray(values["earth_stray_light_ratio"])[~in_field], 0.0
            )
            assert np.isposinf(np.asarray(values["earth_v_mag"])[~in_field]).all()


def test_a_dark_earth_costs_no_integral(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    import time as clock

    rng = np.random.default_rng(6)
    count = 400
    ra = list(rng.uniform(0.0, 360.0, count))
    dec = list(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count))))
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    phase = earth_phase_deg(low_orbit, np.arange(LOW_N))

    def best(row: int) -> float:
        elapsed = []
        for _ in range(3):
            start = clock.perf_counter()
            constraint.in_constraint_batch(low_orbit, ra, dec, indices=[row])
            elapsed.append(clock.perf_counter() - start)
        return min(elapsed)

    # In eclipse every refinement level used to run: forty times the daylight cost.
    assert best(int(np.argmax(phase))) < 2.0 * best(int(np.argmin(phase))) + 0.02


def test_the_sun_behind_the_earth_contributes_nothing(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    """In eclipse, pointing just above the limb toward the hidden Sun."""
    rows = np.arange(LOW_N)
    bodies = bodies_at(low_orbit, rows)
    from_nadir = np.degrees(
        np.arccos(np.sum(unit(bodies["sun"]) * unit(bodies["earth"]), axis=1))
    )
    row = int(np.argmin(np.abs(from_nadir - 30.0)))
    assert from_nadir[row] < 35.0
    time = low_orbit.timestamp[row]
    assert re.EclipseConstraint().in_constraint(time, low_orbit, 0.0, 0.0)
    ra, dec = radec_of(
        clear_of_the_earth(bodies["earth"][row], bodies["sun"][row], 12.0)
    )
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    result = evaluate(constraint, low_orbit, ra, dec, [row])
    values = result.constraint_values
    status = re.StrayLightStatus
    assert values["sun_status"][0] == status.OCCULTED
    assert status.OCCULTED.is_valid
    assert values["sun_stray_light_ratio"][0] == 0.0
    assert values["sun_lit_fraction"][0] == 0.0
    # The Sun's own magnitude is still reported.
    assert values["sun_v_mag"][0] == pytest.approx(-26.74, abs=0.05)
    assert values["earth_status"][0] == status.DARK
    assert values["earth_stray_light_ratio"][0] == 0.0
    assert values["stray_light_ratio"][0] < 1e-3
    assert not bool(result.constraint_array[0])
    # The mask alone agrees.
    assert not constraint.in_constraint(time, low_orbit, ra, dec)


def test_a_setting_sun_is_hidden_part_by_part(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    """Across the limb the Sun's contribution falls with the part in view."""
    rows = np.arange(LOW_N)
    bodies = bodies_at(low_orbit, rows)
    hidden = np.array(
        [
            Reference.hidden("sun", bodies["sun"][row], bodies["earth"][row])
            for row in rows
        ]
    )
    partly = np.flatnonzero(hidden == "partly")
    # Sunset and sunrise, a few seconds each.
    assert 4 <= len(partly) <= 20
    first = (
        partly[np.flatnonzero(np.diff(partly) > 1)[0] + 1 :]
        if (np.diff(partly) > 1).any()
        else partly
    )
    crossing = np.arange(first[0] - 1, first[-1] + 2)
    # One pointing, clear of everything for the whole crossing.
    middle = int(crossing[len(crossing) // 2])
    b = -unit(unit(bodies["earth"][middle]) + 0.5 * unit(bodies["sun"][middle]))
    ra, dec = radec_of(b)
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    n = len(crossing)
    result = evaluate(constraint, low_orbit, [ra] * n, [dec] * n, crossing)
    values = {k: np.asarray(v) for k, v in result.constraint_values.items()}
    assert_values_match(
        values,
        reference_values(
            constraint, low_orbit, np.full(n, ra), np.full(n, dec), crossing
        ),
    )
    fraction = values["sun_lit_fraction"]
    # From one side of the limb to the other, monotonically.
    assert {fraction[0], fraction[-1]} == {0.0, 1.0}
    step = np.diff(fraction)
    assert (step >= 0).all() or (step <= 0).all()
    inside = (fraction > 0) & (fraction < 1)
    assert inside.sum() >= 2
    assert (values["sun_status"][inside] == re.StrayLightStatus.OK).all()
    # The table hardly changes across half a degree: the light follows the
    # fraction of the disk in view.
    whole = values["sun_stray_light_ratio"][fraction == 1.0][0]
    np.testing.assert_allclose(
        values["sun_stray_light_ratio"][inside] / whole, fraction[inside], rtol=0.05
    )


def test_the_moon_behind_the_earth_contributes_nothing(
    table: PointSourceTransmittance, low_orbit: Any
) -> None:
    rows = np.arange(0, LOW_N, 25)
    bodies = bodies_at(low_orbit, rows)
    hidden = np.array(
        [
            Reference.hidden("moon", bodies["moon"][i], bodies["earth"][i])
            for i in range(len(rows))
        ]
    )
    assert (hidden == "all").any() and (hidden == "none").any()
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    # Away from the Earth, so that the pointing itself is never the question.
    for kind in ("all", "none"):
        i = int(np.flatnonzero(hidden == kind)[0])
        ra, dec = radec_of(-bodies["earth"][i])
        values = evaluate(
            constraint, low_orbit, ra, dec, [int(rows[i])]
        ).constraint_values
        occulted = values["moon_status"][0] == re.StrayLightStatus.OCCULTED
        assert occulted == (kind == "all")
        assert (values["moon_stray_light_ratio"][0] == 0.0) == (kind == "all")


def test_illumination_follows_the_distance_of_the_sun(
    table: PointSourceTransmittance, ephem: Any
) -> None:
    """The Earth and Moon are lit by the Sun where it is, not at 1 au: the
    reference applies the inverse square of the Sun's distance to both."""
    row = 300
    bodies = bodies_at(ephem, [row])
    sun_au = np.linalg.norm(bodies["sun"][0] - bodies["earth"][0]) / AU_KM
    # Early March: the Earth is inside 1 au.
    assert 0.985 < sun_au < 0.995
    ra, dec = radec_of(-bodies["sun"][0])
    constraint = StrayLightConstraint(pst=table, conversion_factor=CONVERSION)
    values = evaluate(constraint, ephem, ra, dec, [row]).constraint_values
    expected = reference_values(
        constraint, ephem, np.array([ra]), np.array([dec]), [row]
    )
    for body in ("earth", "moon"):
        assert values[f"{body}_v_mag"][0] == pytest.approx(
            expected[f"{body}_v_mag"][0], abs=1e-9
        )
    # The Moon's magnitude carries the term explicitly.
    moon = bodies["moon"][0]
    phase = angle_deg(bodies["sun"][0] - moon, -moon)
    moon_sun_au = np.linalg.norm(bodies["sun"][0] - moon) / AU_KM
    at_1au = (
        constraint.moon_v_mag_full
        + 5 * np.log10(np.linalg.norm(moon) / 384400.0)
        + 0.026 * phase
        + 4e-9 * phase**4
    )
    assert values["moon_v_mag"][0] - at_1au == pytest.approx(
        5 * np.log10(moon_sun_au), abs=1e-9
    )


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
    kwargs: dict[str, Any] = {
        "pst": table,
        "conversion_factor": CONVERSION,
        "sun_azimuth_deg": 180,
    }
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
    kwargs: dict[str, Any] = {"times": times, "target_rolls": rolls}
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


# --- free roll ---------------------------------------------------------------

SWEPT = 72
SWEPT_ROLLS = np.arange(SWEPT) * (360.0 / SWEPT)


@pytest.fixture(scope="module")
def free_roll_case(
    ephem: Any, sample: tuple[Array, Array, Any]
) -> tuple[StrayLightConstraint, Array, Array, list[Any], Array]:
    """A stray-light rule whose answer depends on the roll at most pointings:
    (rule, ra, dec, times, violated at each of SWEPT_ROLLS x pointing)."""
    ra, dec, rows = (a[:48] for a in sample)
    times = [ephem.timestamp[int(i)] for i in rows]
    pst = make_table(symmetric=False)
    probe = StrayLightConstraint(pst=pst, conversion_factor=CONVERSION)
    nominal = np.asarray(
        probe.evaluate_elementwise(ephem, ra, dec, times=times).constraint_values[
            "stray_light_ratio"
        ]
    )
    stray = StrayLightConstraint(
        pst=pst,
        conversion_factor=CONVERSION,
        max_ratio=float(np.nanmedian(nominal)),
    )
    return stray, ra, dec, times, at_each_roll(stray, ephem, ra, dec, times)


def at_each_roll(
    tree: Any, ephem: Any, ra: Array, dec: Array, times: list[Any]
) -> Array:
    """The mask at each of SWEPT_ROLLS given explicitly: (rolls, pointings)."""
    return np.array(
        [
            tree.in_constraint_elementwise(
                ephem, ra, dec, times=times, target_rolls=[float(roll)] * len(ra)
            )
            for roll in SWEPT_ROLLS
        ]
    )


def free_roll_trees(stray: StrayLightConstraint) -> dict[str, Any]:
    """Trees in which another rule needs a roll, so a free roll is swept."""
    offset = re.SunConstraint(min_angle=100).boresight_offset(pitch_deg=50)
    solar = re.SolarRollConstraint(tolerance_deg=40.0)
    sun = re.SunConstraint(min_angle=60)
    return {
        "or_offset": offset | stray,
        "and_offset": offset & stray,
        "or_solar_roll": solar | stray,
        "nested": (stray & sun) | offset,
        "negated": ~stray | offset,
        "xor": stray ^ offset,
        "wrapped": stray.boresight_offset(pitch_deg=0.0) | offset,
    }


@pytest.mark.parametrize(
    "name",
    ["or_offset", "and_offset", "or_solar_roll", "nested", "negated", "xor", "wrapped"],
)
def test_free_roll_holds_every_rule_at_one_roll(
    free_roll_case: tuple[StrayLightConstraint, Array, Array, list[Any], Array],
    ephem: Any,
    name: str,
) -> None:
    """With no roll given, a tree that needs one is satisfied exactly where
    some roll satisfies all of it: the stray-light rule is evaluated at each
    candidate with the others, not left at its nominal attitude."""
    stray, ra, dec, times, _ = free_roll_case
    tree = free_roll_trees(stray)[name]
    explicit = at_each_roll(tree, ephem, ra, dec, times)
    expected: npt.NDArray[np.bool_] = np.asarray(explicit.all(axis=0))
    free: dict[str, Any] = {"n_roll_samples": SWEPT}
    np.testing.assert_array_equal(
        tree.in_constraint_elementwise(ephem, ra, dec, times=times, **free), expected
    )
    np.testing.assert_array_equal(
        tree.evaluate_elementwise(ephem, ra, dec, times=times, **free).constraint_array,
        expected,
    )
    for i in range(0, len(ra), 4):
        point = (float(ra[i]), float(dec[i]))
        want = bool(expected[i])
        assert tree.in_constraint(times[i], ephem, *point, **free) == want
        single = tree.evaluate(ephem, *point, times=[times[i]], **free)
        assert bool(single.constraint_array[0]) == want
        assert single.all_satisfied == (not want)
        grid = tree.in_constraint_batch(
            ephem, [point[0]], [point[1]], times=[times[i]], **free
        )
        assert bool(grid[0, 0]) == want
        (batch,) = tree.evaluate_batch(
            ephem, [point[0]], [point[1]], times=[times[i]], **free
        )
        assert bool(batch.constraint_array[0]) == want
        # No roll range exactly where no roll satisfies the tree.
        ranges = tree.roll_range(times[i], ephem, *point, n_roll_samples=SWEPT)
        assert (len(ranges) == 0) == want
    if name in ("or_offset", "and_offset", "nested"):
        # The case is not vacuous: both answers occur, and the stray-light
        # rule at its nominal attitude would give a different mask.
        assert expected.any() and not expected.all()


def test_a_tree_no_roll_satisfies_is_violated_though_the_nominal_roll_passes(
    ephem: Any, sample: tuple[Array, Array, Any]
) -> None:
    """The stray-light rule passes at its nominal attitude and over a narrow
    range of rolls; an offset rule passes over another range; no roll passes
    both. The tree is violated, and has no roll range."""
    ra, dec, rows = (a[:48] for a in sample)
    times = [ephem.timestamp[int(i)] for i in rows]
    pst = make_table(symmetric=False)
    probe = StrayLightConstraint(pst=pst, conversion_factor=CONVERSION)

    def ratios(**kwargs: Any) -> Array:
        return np.asarray(
            probe.evaluate_elementwise(
                ephem, ra, dec, times=times, **kwargs
            ).constraint_values["stray_light_ratio"]
        )

    nominal = ratios()
    by_roll = np.array(
        [ratios(target_rolls=[float(roll)] * len(ra)) for roll in SWEPT_ROLLS]
    )
    offsets = [
        re.SunConstraint(min_angle=angle).boresight_offset(pitch_deg=pitch)
        for pitch in (30.0, 50.0, 70.0)
        for angle in (60.0, 80.0, 100.0, 120.0, 140.0)
    ]
    offset_masks = [at_each_roll(offset, ephem, ra, dec, times) for offset in offsets]
    found = 0
    for i in range(len(ra)):
        if not np.isfinite(by_roll[:, i]).all():
            continue
        # A limit the nominal attitude meets and most rolls do not.
        limit = max(1.02 * nominal[i], float(np.quantile(by_roll[:, i], 0.1)))
        stray_violated = by_roll[:, i] > limit
        if stray_violated.all() or not stray_violated.any():
            continue
        for offset, mask in zip(offsets, offset_masks):
            if mask[:, i].all() or not (stray_violated | mask[:, i]).all():
                continue
            stray = StrayLightConstraint(
                pst=pst, conversion_factor=CONVERSION, max_ratio=limit
            )
            tree = offset | stray
            point = (float(ra[i]), float(dec[i]))
            free: dict[str, Any] = {"n_roll_samples": SWEPT}
            # Each rule alone can be satisfied; stray light is at its nominal roll.
            assert not stray.in_constraint(times[i], ephem, *point)
            assert not offset.in_constraint(times[i], ephem, *point, **free)
            assert tree.in_constraint(times[i], ephem, *point, **free)
            result = tree.evaluate(ephem, *point, times=[times[i]], **free)
            assert not result.all_satisfied
            assert tree.roll_range(times[i], ephem, *point, n_roll_samples=SWEPT) == []
            assert stray.roll_range(times[i], ephem, *point, n_roll_samples=SWEPT)
            assert offset.roll_range(times[i], ephem, *point, n_roll_samples=SWEPT)
            found += 1
            break
    assert found >= 3


def test_without_a_rule_that_needs_a_roll_the_attitude_is_nominal(
    free_roll_case: tuple[StrayLightConstraint, Array, Array, list[Any], Array],
    ephem: Any,
) -> None:
    """Alone, or with rules that ignore the roll, nothing is swept."""
    stray, ra, dec, times, by_roll = free_roll_case
    values = stray.evaluate_elementwise(ephem, ra, dec, times=times).constraint_values
    ratio = np.asarray(values["stray_light_ratio"])
    nominal = ~(ratio <= stray.max_ratio)
    np.testing.assert_array_equal(
        stray.in_constraint_elementwise(ephem, ra, dec, times=times), nominal
    )
    # Not the same thing as "violated at every roll".
    assert (nominal != by_roll.all(axis=0)).any()
    sun = re.SunConstraint(min_angle=60)
    sun_mask = np.asarray(sun.in_constraint_elementwise(ephem, ra, dec, times=times))
    np.testing.assert_array_equal(
        (stray | sun).in_constraint_elementwise(ephem, ra, dec, times=times),
        nominal | sun_mask,
    )
    grid = (stray | sun).in_constraint_batch(
        ephem, list(ra[:6]), list(dec[:6]), times=times[:6]
    )
    np.testing.assert_array_equal(np.diag(grid), (nominal | sun_mask)[:6])


def test_field_of_regard_of_a_swept_tree(
    free_roll_case: tuple[StrayLightConstraint, Array, Array, list[Any], Array],
    ephem: Any,
) -> None:
    """A rule can only take sky away: with stray light the field of regard of
    a swept tree is no larger, and is the same whichever way it is nested."""
    stray, *_ = free_roll_case
    offset = re.SunConstraint(min_angle=100).boresight_offset(pitch_deg=50)
    kwargs: dict[str, Any] = {"index": 300, "n_points": 600, "n_roll_samples": 24}
    alone = offset.instantaneous_field_of_regard(ephem, **kwargs)
    together = (offset | stray).instantaneous_field_of_regard(ephem, **kwargs)
    assert 0.0 < together < alone
    either = (offset & stray).instantaneous_field_of_regard(ephem, **kwargs)
    assert either >= alone
    sun = re.SunConstraint(min_angle=20)
    nested = ((offset | stray) | sun).instantaneous_field_of_regard(ephem, **kwargs)
    flat = (offset | stray | sun).instantaneous_field_of_regard(ephem, **kwargs)
    assert nested == pytest.approx(flat, rel=1e-12)
    assert nested <= together


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
            ({"off_axis_deg": [0.0]}, "at least two"),
            ({"off_axis_deg": [0.0, 10.0, 5.0]}, "strictly increasing"),
            ({"off_axis_deg": [-5.0, 10.0, 20.0]}, r"\[0, 180\]"),
            ({"azimuth_deg": [0.0, 200.0]}, "symmetric table"),
            ({"azimuth_deg": [90.0, 0.0]}, "strictly increasing"),
            (
                {"azimuth_deg": [0.0, 400.0], "azimuth_symmetric": False},
                "less than 360",
            ),
            ({"log10_pst": [[-1.0, -2.0], [-2.0, -3.0]]}, "shape"),
            ({"log10_pst": [[None, None]] * 3}, "no finite value"),
        ],
    )
    def test_rejects_malformed_tables(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        base: dict[str, Any] = {
            "off_axis_deg": [0.0, 10.0, 20.0],
            "azimuth_deg": [0.0, 180.0],
            "log10_pst": [[-1.0, -2.0], [-2.0, -3.0], [-3.0, -4.0]],
        }
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
            {"conversion_factor": 0.0},
            {"conversion_factor": float("nan")},
            {"max_ratio": -1.0},
            {"in_field_deg": -0.1},
            {"earth_albedo": 0.0},
            {"zodi": "sky"},
            {"roll_offset_deg": float("inf")},
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
