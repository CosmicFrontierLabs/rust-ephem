"""Conversions between Cartesian state vectors and orbital elements."""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tle import WGS72_EARTH_MU_M3_S2

__all__ = [
    "OsculatingElements",
    "WGS72_EARTH_MU_KM3_S2",
    "osculating_elements_from_state",
]

WGS72_EARTH_MU_KM3_S2 = WGS72_EARTH_MU_M3_S2 / 1.0e9

_SINGULARITY_TOLERANCE = 1.0e-10
_PARABOLIC_TOLERANCE = 1.0e-12


class OsculatingElements(TypedDict):
    """Instantaneous classical orbital elements derived from a Cartesian state."""

    semimajor_axis_km: float
    eccentricity: float
    inclination_deg: float
    right_ascension_of_ascending_node_deg: float
    argument_of_periapsis_deg: float
    true_anomaly_deg: float
    gravitational_parameter_km3_s2: float


def _normalize_degrees(angle_rad: float) -> float:
    angle_deg = math.degrees(angle_rad) % 360.0
    if math.isclose(angle_deg, 360.0, rel_tol=0.0, abs_tol=1.0e-12):
        return 0.0
    return angle_deg


def _oriented_angle(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> float:
    """Return the oriented angle from first to second about normal."""
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    normal_norm = float(np.linalg.norm(normal))
    sine = float(np.dot(np.cross(first, second), normal))
    sine /= first_norm * second_norm * normal_norm
    cosine = float(np.dot(first, second)) / (first_norm * second_norm)
    return math.atan2(sine, cosine)


def osculating_elements_from_state(
    position_km: ArrayLike,
    velocity_km_s: ArrayLike,
    mu_km3_s2: float = WGS72_EARTH_MU_KM3_S2,
) -> OsculatingElements:
    """Derive instantaneous classical osculating elements from one state vector.

    ``position_km`` and ``velocity_km_s`` must describe the same epoch in the
    same central-body-centered inertial Cartesian frame. The returned
    orientation angles are relative to that frame and normalized to the range
    [0, 360) degrees. The epoch and frame are intentionally not inferred or
    included in the result; callers serializing the elements should store both
    alongside them.

    Singular classical-element cases use deterministic conventions:

    * Circular inclined orbit: argument of periapsis is zero and true anomaly
      is the argument of latitude.
    * Eccentric equatorial orbit: right ascension of the ascending node is zero
      and argument of periapsis is the longitude of periapsis.
    * Circular equatorial orbit: both undefined orientation angles are zero and
      true anomaly is the true longitude.

    Parameters
    ----------
    position_km:
        Three-dimensional position vector in kilometers.
    velocity_km_s:
        Three-dimensional velocity vector in kilometers per second.
    mu_km3_s2:
        Central-body gravitational parameter in km^3/s^2. Defaults to the
        WGS-72 Earth value used by SGP4.

    Returns
    -------
    OsculatingElements
        Semimajor axis, eccentricity, inclination, right ascension of the
        ascending node, argument of periapsis, true anomaly, and the
        gravitational parameter used.

    Raises
    ------
    ValueError
        If either vector is malformed or non-finite, the gravitational
        parameter is invalid, the state has no defined orbital plane, or the
        state is parabolic to numerical precision.
    """
    position = np.asarray(position_km, dtype=np.float64)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)

    if position.shape != (3,):
        raise ValueError("position_km must be a three-dimensional vector")
    if velocity.shape != (3,):
        raise ValueError("velocity_km_s must be a three-dimensional vector")
    if not np.all(np.isfinite(position)):
        raise ValueError("position_km must contain only finite values")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_km_s must contain only finite values")

    mu = float(mu_km3_s2)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu_km3_s2 must be finite and greater than zero")

    radius = float(np.linalg.norm(position))
    if radius == 0.0:
        raise ValueError("position_km must have non-zero magnitude")

    speed = float(np.linalg.norm(velocity))
    angular_momentum = np.cross(position, velocity)
    angular_momentum_magnitude = float(np.linalg.norm(angular_momentum))
    if angular_momentum_magnitude <= np.finfo(np.float64).eps * radius * speed:
        raise ValueError("state must define a non-degenerate orbital plane")

    node = np.array(
        [-angular_momentum[1], angular_momentum[0], 0.0],
        dtype=np.float64,
    )
    node_magnitude = float(np.linalg.norm(node))
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu - position / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))

    kinetic_energy = speed * speed / 2.0
    potential_magnitude = mu / radius
    specific_energy = kinetic_energy - potential_magnitude
    energy_scale = max(kinetic_energy, potential_magnitude)
    if abs(specific_energy) <= _PARABOLIC_TOLERANCE * energy_scale:
        raise ValueError("parabolic states do not have a finite semimajor axis")
    semimajor_axis = -mu / (2.0 * specific_energy)

    inclination = math.atan2(
        node_magnitude,
        float(angular_momentum[2]),
    )
    circular = eccentricity <= _SINGULARITY_TOLERANCE
    equatorial = node_magnitude <= _SINGULARITY_TOLERANCE * angular_momentum_magnitude

    if equatorial:
        right_ascension = 0.0
    else:
        right_ascension = math.atan2(float(node[1]), float(node[0]))

    if circular and equatorial:
        argument_of_periapsis = 0.0
        direction = math.copysign(1.0, float(angular_momentum[2]))
        true_anomaly = math.atan2(
            direction * float(position[1]),
            float(position[0]),
        )
    elif circular:
        argument_of_periapsis = 0.0
        true_anomaly = _oriented_angle(node, position, angular_momentum)
    elif equatorial:
        direction = math.copysign(1.0, float(angular_momentum[2]))
        argument_of_periapsis = math.atan2(
            direction * float(eccentricity_vector[1]),
            float(eccentricity_vector[0]),
        )
        true_anomaly = _oriented_angle(
            eccentricity_vector,
            position,
            angular_momentum,
        )
    else:
        argument_of_periapsis = _oriented_angle(
            node,
            eccentricity_vector,
            angular_momentum,
        )
        true_anomaly = _oriented_angle(
            eccentricity_vector,
            position,
            angular_momentum,
        )

    return OsculatingElements(
        semimajor_axis_km=float(semimajor_axis),
        eccentricity=eccentricity,
        inclination_deg=math.degrees(inclination),
        right_ascension_of_ascending_node_deg=_normalize_degrees(right_ascension),
        argument_of_periapsis_deg=_normalize_degrees(argument_of_periapsis),
        true_anomaly_deg=_normalize_degrees(true_anomaly),
        gravitational_parameter_km3_s2=mu,
    )
