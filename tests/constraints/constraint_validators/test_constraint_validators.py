import pytest

from rust_ephem.constraints import AirmassConstraint, MoonPhaseConstraint


class TestValidators:
    def test_airmass_raises_on_invalid(self) -> None:
        with pytest.raises(ValueError):
            AirmassConstraint(min_airmass=2.0, max_airmass=1.5)

    def test_airmass_max_airmass(self) -> None:
        valid_airmass: AirmassConstraint = AirmassConstraint(
            min_airmass=1.0, max_airmass=2.0
        )
        assert valid_airmass.max_airmass == 2.0

    def test_moon_phase_raises_on_invalid_illumination(self) -> None:
        with pytest.raises(ValueError):
            MoonPhaseConstraint(
                min_illumination=0.8, max_illumination=0.5, max_distance=1.0
            )

    def test_moon_phase_raises_on_invalid_distance(self) -> None:
        with pytest.raises(ValueError):
            MoonPhaseConstraint(
                min_distance=5.0, max_distance=4.0, max_illumination=0.9
            )

    def test_moon_phase_max_distance(self) -> None:
        valid_phase: MoonPhaseConstraint = MoonPhaseConstraint(
            max_illumination=0.9, min_distance=1.0, max_distance=2.0
        )
        assert valid_phase.max_distance == 2.0
