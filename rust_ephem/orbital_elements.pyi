from typing import TypedDict

from numpy.typing import ArrayLike

WGS72_EARTH_MU_KM3_S2: float

class OsculatingElements(TypedDict):
    semimajor_axis_km: float
    eccentricity: float
    inclination_deg: float
    right_ascension_of_ascending_node_deg: float
    argument_of_periapsis_deg: float
    true_anomaly_deg: float
    gravitational_parameter_km3_s2: float

def osculating_elements_from_state(
    position_km: ArrayLike,
    velocity_km_s: ArrayLike,
    mu_km3_s2: float = ...,
) -> OsculatingElements: ...
