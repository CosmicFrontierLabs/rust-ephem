import math

import numpy as np
import pytest

from rust_ephem import (
    WGS72_EARTH_MU_KM3_S2,
    osculating_elements_from_state,
)


class TestOsculatingElementsFromState:
    def test_textbook_elliptical_state(self) -> None:
        elements = osculating_elements_from_state(
            position_km=[-6045.0, -3490.0, 2500.0],
            velocity_km_s=[-3.457, 6.618, 2.533],
            mu_km3_s2=398600.0,
        )

        assert elements["semimajor_axis_km"] == pytest.approx(
            8788.095,
            abs=0.001,
        )
        assert elements["eccentricity"] == pytest.approx(0.171212, abs=1.0e-6)
        assert elements["inclination_deg"] == pytest.approx(153.249, abs=0.001)
        assert elements["right_ascension_of_ascending_node_deg"] == pytest.approx(
            255.279, abs=0.001
        )
        assert elements["argument_of_periapsis_deg"] == pytest.approx(
            20.068,
            abs=0.001,
        )
        assert elements["true_anomaly_deg"] == pytest.approx(
            28.446,
            abs=0.001,
        )
        assert elements["gravitational_parameter_km3_s2"] == 398600.0

    def test_circular_equatorial_uses_true_longitude(self) -> None:
        radius_km = 7000.0
        longitude_deg = 75.0
        longitude_rad = math.radians(longitude_deg)
        circular_speed = math.sqrt(WGS72_EARTH_MU_KM3_S2 / radius_km)
        position = radius_km * np.array(
            [math.cos(longitude_rad), math.sin(longitude_rad), 0.0]
        )
        velocity = circular_speed * np.array(
            [-math.sin(longitude_rad), math.cos(longitude_rad), 0.0]
        )

        elements = osculating_elements_from_state(position, velocity)

        assert elements["semimajor_axis_km"] == pytest.approx(radius_km)
        assert elements["eccentricity"] == pytest.approx(0.0, abs=1.0e-14)
        assert elements["inclination_deg"] == pytest.approx(0.0)
        assert elements["right_ascension_of_ascending_node_deg"] == 0.0
        assert elements["argument_of_periapsis_deg"] == 0.0
        assert elements["true_anomaly_deg"] == pytest.approx(longitude_deg)

    def test_circular_inclined_uses_argument_of_latitude(self) -> None:
        radius_km = 7000.0
        inclination_rad = math.radians(30.0)
        circular_speed = math.sqrt(WGS72_EARTH_MU_KM3_S2 / radius_km)
        position = np.array([radius_km, 0.0, 0.0])
        velocity = circular_speed * np.array(
            [0.0, math.cos(inclination_rad), math.sin(inclination_rad)]
        )

        elements = osculating_elements_from_state(position, velocity)

        assert elements["eccentricity"] == pytest.approx(0.0, abs=1.0e-14)
        assert elements["inclination_deg"] == pytest.approx(30.0)
        assert elements["right_ascension_of_ascending_node_deg"] == pytest.approx(0.0)
        assert elements["argument_of_periapsis_deg"] == 0.0
        assert elements["true_anomaly_deg"] == pytest.approx(0.0)

    def test_eccentric_equatorial_uses_longitude_of_periapsis(self) -> None:
        semimajor_axis_km = 10_000.0
        eccentricity = 0.2
        periapsis_km = semimajor_axis_km * (1.0 - eccentricity)
        speed_km_s = math.sqrt(
            WGS72_EARTH_MU_KM3_S2 * (2.0 / periapsis_km - 1.0 / semimajor_axis_km)
        )
        longitude_rad = math.radians(40.0)
        position = periapsis_km * np.array(
            [math.cos(longitude_rad), math.sin(longitude_rad), 0.0]
        )
        velocity = speed_km_s * np.array(
            [-math.sin(longitude_rad), math.cos(longitude_rad), 0.0]
        )

        elements = osculating_elements_from_state(position, velocity)

        assert elements["semimajor_axis_km"] == pytest.approx(semimajor_axis_km)
        assert elements["eccentricity"] == pytest.approx(eccentricity)
        assert elements["right_ascension_of_ascending_node_deg"] == 0.0
        assert elements["argument_of_periapsis_deg"] == pytest.approx(40.0)
        assert elements["true_anomaly_deg"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("position", "velocity", "message"),
        [
            ([1.0, 2.0], [1.0, 2.0, 3.0], "position_km"),
            ([1.0, 2.0, 3.0], [1.0, 2.0], "velocity_km_s"),
            ([math.nan, 0.0, 0.0], [0.0, 1.0, 0.0], "position_km"),
            ([1.0, 0.0, 0.0], [0.0, math.inf, 0.0], "velocity_km_s"),
        ],
    )
    def test_rejects_malformed_states(
        self,
        position: list[float],
        velocity: list[float],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            osculating_elements_from_state(position, velocity)

    @pytest.mark.parametrize("mu", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_invalid_gravitational_parameter(self, mu: float) -> None:
        with pytest.raises(ValueError, match="mu_km3_s2"):
            osculating_elements_from_state(
                [7000.0, 0.0, 0.0],
                [0.0, 7.5, 0.0],
                mu,
            )

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="non-zero magnitude"):
            osculating_elements_from_state(
                [0.0, 0.0, 0.0],
                [0.0, 7.5, 0.0],
            )

    def test_rejects_radial_state(self) -> None:
        with pytest.raises(ValueError, match="orbital plane"):
            osculating_elements_from_state(
                [7000.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )

    def test_rejects_parabolic_state(self) -> None:
        radius_km = 7000.0
        escape_speed = math.sqrt(2.0 * WGS72_EARTH_MU_KM3_S2 / radius_km)

        with pytest.raises(ValueError, match="parabolic"):
            osculating_elements_from_state(
                [radius_km, 0.0, 0.0],
                [0.0, escape_speed, 0.0],
            )
