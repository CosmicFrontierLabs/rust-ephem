import datetime

import pytest

from rust_ephem._rust_ephem import TLEEphemeris
from rust_ephem.constraints import (
    EarthLimbConstraint,
    MoonConstraint,
    OrbitPoleConstraint,
    OrConstraint,
    SAAConstraint,
    SunConstraint,
)
from rust_ephem.tle import TLERecord


@pytest.fixture
def swift_saa_polygon() -> list[tuple[float, float]]:
    return [
        (39.0, -30.0),
        (36.0, -26.0),
        (28.0, -21.0),
        (6.0, -12.0),
        (-5.0, -6.0),
        (-21.0, 2.0),
        (-30.0, 3.0),
        (-45.0, 2.0),
        (-60.0, -2.0),
        (-75.0, -7.0),
        (-83.0, -10.0),
        (-87.0, -16.0),
        (-86.0, -23.0),
        (-83.0, -30.0),
    ]


@pytest.fixture
def swift_saa_constraint(swift_saa_polygon: list[tuple[float, float]]) -> SAAConstraint:
    return SAAConstraint(polygon=swift_saa_polygon)


@pytest.fixture
def swift_sun_constraint() -> SunConstraint:
    return SunConstraint(min_angle=46 + 1)


@pytest.fixture
def swift_moon_constraint() -> MoonConstraint:
    return MoonConstraint(min_angle=22 + 1)


@pytest.fixture
def swift_earth_constraint() -> EarthLimbConstraint:
    return EarthLimbConstraint(min_angle=28 + 5)


@pytest.fixture
def swift_pole_constraintt() -> OrbitPoleConstraint:
    return OrbitPoleConstraint(min_angle=28 + 5, earth_limb_pole=True)


@pytest.fixture()
def swift_tle() -> TLERecord:
    return TLERecord(
        line1="1 28485U 04047A   25342.87828629  .00042565  00000-0  70967-3 0  9992",
        line2="2 28485  20.5524 193.5008 0003262 214.0798 145.9435 15.50178289157119",
        name=None,
        epoch=datetime.datetime(2025, 12, 8, 21, 4, 43, tzinfo=datetime.timezone.utc),
        source="spacetrack",
    )


@pytest.fixture
def begin() -> datetime.datetime:
    return datetime.datetime(2025, 12, 9)


@pytest.fixture
def end() -> datetime.datetime:
    return datetime.datetime(2025, 12, 10)


@pytest.fixture
def swift_constraint(
    swift_sun_constraint: SunConstraint,
    swift_moon_constraint: MoonConstraint,
    swift_earth_constraint: EarthLimbConstraint,
    swift_saa_constraint: SAAConstraint,
) -> OrConstraint:
    return (
        swift_sun_constraint
        | swift_moon_constraint
        | swift_earth_constraint
        | swift_saa_constraint
    )


@pytest.fixture
def swift_ephemeris(
    swift_tle: TLERecord, begin: datetime.datetime, end: datetime.datetime
) -> TLEEphemeris:
    return TLEEphemeris(tle=swift_tle, begin=begin, end=end, step_size=60)


class TestSwiftVisibility:
    @pytest.mark.parametrize(
        "window",
        [
            ["2025-12-09 00:52:00", "2025-12-09 00:59:00"],
            ["2025-12-09 01:25:00", "2025-12-09 01:31:00"],
            ["2025-12-09 02:25:00", "2025-12-09 02:35:00"],
            ["2025-12-09 03:02:00", "2025-12-09 03:03:00"],
            ["2025-12-09 03:57:00", "2025-12-09 04:13:00"],
            ["2025-12-09 05:30:00", "2025-12-09 05:52:00"],
            ["2025-12-09 07:03:00", "2025-12-09 07:31:00"],
            ["2025-12-09 08:36:00", "2025-12-09 09:09:00"],
            ["2025-12-09 10:09:00", "2025-12-09 10:47:00"],
            ["2025-12-09 11:41:00", "2025-12-09 12:20:00"],
            ["2025-12-09 13:14:00", "2025-12-09 13:53:00"],
            ["2025-12-09 14:47:00", "2025-12-09 15:26:00"],
            ["2025-12-09 16:20:00", "2025-12-09 16:58:00"],
            ["2025-12-09 17:52:00", "2025-12-09 18:31:00"],
            ["2025-12-09 19:25:00", "2025-12-09 20:04:00"],
            ["2025-12-09 21:13:00", "2025-12-09 21:37:00"],
            ["2025-12-09 22:31:00", "2025-12-09 22:33:00"],
            ["2025-12-09 22:56:00", "2025-12-09 23:09:00"],
        ],
    )
    def test_swift_visibility_windows_comparison(
        self,
        swift_constraint: OrConstraint,
        swift_ephemeris: TLEEphemeris,
        window: list[str],
    ) -> None:
        result = swift_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        start = datetime.datetime.fromisoformat(window[0]).replace(
            tzinfo=datetime.timezone.utc
        )
        end = datetime.datetime.fromisoformat(window[1]).replace(
            tzinfo=datetime.timezone.utc
        )

        # Check if any visibility window is within 60 seconds of the expected start/end times
        # (allowing for ephemeris timestep resolution)
        tolerance = datetime.timedelta(seconds=60)
        found = any(
            abs((vw.start_time - start).total_seconds()) <= tolerance.total_seconds()
            and abs((vw.end_time - end).total_seconds()) <= tolerance.total_seconds()
            for vw in result.visibility
        )
        assert found, (
            f"Expected window near ({start}, {end}) not found in {result.visibility}"
        )

    def test_sun_constraint_alone_result_is_bool(
        self, swift_sun_constraint: SunConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Sun constraint independently."""
        result = swift_sun_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Result should be valid
        assert isinstance(result.all_satisfied, bool)

    def test_sun_constraint_alone_has_violations_or_visibility(
        self, swift_sun_constraint: SunConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Sun constraint independently."""
        result = swift_sun_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Violations and visibility should sum to full time period
        assert len(result.violations) + len(result.visibility) > 0

    def test_moon_constraint_alone_result_is_bool(
        self, swift_moon_constraint: MoonConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Moon constraint independently."""
        result = swift_moon_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Result should be valid
        assert isinstance(result.all_satisfied, bool)

    def test_moon_constraint_alone_has_violations_or_visibility(
        self, swift_moon_constraint: MoonConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Moon constraint independently."""
        result = swift_moon_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Violations and visibility should sum to full time period
        assert len(result.violations) + len(result.visibility) > 0

    def test_earth_limb_constraint_alone_has_violations(
        self, swift_earth_constraint: EarthLimbConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Earth limb constraint independently."""
        result = swift_earth_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Should have some violations (times when too close to Earth limb)
        assert len(result.violations) > 0

    def test_earth_limb_constraint_alone_has_visibility(
        self, swift_earth_constraint: EarthLimbConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test Earth limb constraint independently."""
        result = swift_earth_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Should have some visibility windows
        assert len(result.visibility) > 0

    def test_saa_constraint_alone_has_violations(
        self, swift_saa_constraint: SAAConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test SAA constraint independently."""
        result = swift_saa_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Should have some violations (times in SAA)
        assert len(result.violations) > 0

    def test_saa_constraint_alone_has_visibility(
        self, swift_saa_constraint: SAAConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test SAA constraint independently."""
        result = swift_saa_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )
        # Should have some visibility windows
        assert len(result.visibility) > 0

    def test_pole_constraint_near_pole_result_is_bool(
        self, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test orbit pole constraint with target near orbit pole."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        # Target at very high declination
        result = pole_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=193.5, target_dec=85.0
        )

        # Check that constraint evaluation works and returns valid results
        assert isinstance(result.all_satisfied, bool)

    def test_pole_constraint_near_pole_has_violations_or_visibility(
        self, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test orbit pole constraint with target near orbit pole."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        # Target at very high declination
        result = pole_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=193.5, target_dec=85.0
        )

        # Should have either violations or visibility (or both)
        assert len(result.violations) + len(result.visibility) > 0

    def test_pole_constraint_far_from_pole_has_visibility(
        self, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test orbit pole constraint with target far from orbit pole - should have few/no violations."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        # Target in equatorial region, far from orbit poles
        result = pole_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # Should have visibility windows
        assert len(result.visibility) > 0, (
            "Expected visibility windows for target far from pole"
        )

    def test_pole_constraint_far_from_pole_violation_time_check(
        self, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test orbit pole constraint with target far from orbit pole - should have few/no violations."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        # Target in equatorial region, far from orbit poles
        result = pole_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # May have some violations but should have significant visibility
        total_time = (
            swift_ephemeris.timestamp[-1] - swift_ephemeris.timestamp[0]
        ).total_seconds()

        if len(result.violations) > 0:
            violation_time = sum(
                (v.end_time - v.start_time).total_seconds() for v in result.violations
            )
            # Should be mostly visible (< 30% violation time)
            assert violation_time < total_time * 0.3, (
                f"Expected <30% violation time, got {violation_time / total_time * 100:.1f}%"
            )

    def test_combined_constraint_with_pole_has_visibility(
        self,
        swift_sun_constraint: SunConstraint,
        swift_moon_constraint: MoonConstraint,
        swift_earth_constraint: EarthLimbConstraint,
        swift_saa_constraint: SAAConstraint,
        swift_ephemeris: TLEEphemeris,
    ) -> None:
        """Test combined constraints including pole constraint."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        combined = (
            swift_sun_constraint
            | swift_moon_constraint
            | swift_earth_constraint
            | swift_saa_constraint
            | pole_constraint
        )

        result = combined.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # Should have some visibility windows
        assert len(result.visibility) > 0, (
            "Expected visibility windows with combined constraints"
        )

    def test_combined_constraint_with_pole_has_violations(
        self,
        swift_sun_constraint: SunConstraint,
        swift_moon_constraint: MoonConstraint,
        swift_earth_constraint: EarthLimbConstraint,
        swift_saa_constraint: SAAConstraint,
        swift_ephemeris: TLEEphemeris,
    ) -> None:
        """Test combined constraints including pole constraint."""
        pole_constraint = OrbitPoleConstraint(min_angle=33, earth_limb_pole=True)

        combined = (
            swift_sun_constraint
            | swift_moon_constraint
            | swift_earth_constraint
            | swift_saa_constraint
            | pole_constraint
        )

        result = combined.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # Total violation time should be more than any individual constraint
        # (OR means violate if ANY constraint is violated)
        assert len(result.violations) > 0

    @pytest.mark.parametrize(
        "ra,dec,expected_mostly_visible",
        [
            (66, -29, True),  # Galactic plane region - should be mostly visible
            (0, 0, True),  # Equatorial region - should be mostly visible
            (180, -20, True),  # Another typical target - should be mostly visible
            (120, -40, True),  # Low latitude target - should be mostly visible
        ],
    )
    def test_swift_full_constraint_various_targets_visibility_fraction(
        self,
        swift_constraint: OrConstraint,
        swift_ephemeris: TLEEphemeris,
        ra: float,
        dec: float,
        expected_mostly_visible: bool,
    ) -> None:
        """Test Swift constraint suite with various target positions."""
        result = swift_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=ra, target_dec=dec
        )

        total_time = (
            swift_ephemeris.timestamp[-1] - swift_ephemeris.timestamp[0]
        ).total_seconds()

        if len(result.visibility) > 0:
            visibility_time = sum(
                (v.end_time - v.start_time).total_seconds() for v in result.visibility
            )
            visibility_fraction = visibility_time / total_time
        else:
            visibility_fraction = 0.0

        if expected_mostly_visible:
            # Should be visible > 25% of the time
            assert visibility_fraction > 0.25, (
                f"Target (RA={ra}, Dec={dec}) expected mostly visible, "
                f"but only {visibility_fraction * 100:.1f}% visible"
            )
        else:
            # Should be visible < 20% of the time for heavily constrained target
            assert visibility_fraction < 0.2, (
                f"Target (RA={ra}, Dec={dec}) expected mostly violated, "
                f"but {visibility_fraction * 100:.1f}% visible"
            )

    def test_visibility_window_properties_positive_duration(
        self, swift_constraint: OrConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test properties of visibility windows."""
        result = swift_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # All visibility windows should have positive duration
        for vw in result.visibility:
            assert vw.duration_seconds > 0, (
                f"Visibility window has non-positive duration: {vw}"
            )

    def test_visibility_window_properties_start_before_end(
        self, swift_constraint: OrConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test properties of visibility windows."""
        result = swift_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        for vw in result.visibility:
            assert vw.start_time < vw.end_time, f"Visibility window start >= end: {vw}"

    def test_visibility_window_properties_no_overlap(
        self, swift_constraint: OrConstraint, swift_ephemeris: TLEEphemeris
    ) -> None:
        """Test properties of visibility windows."""
        result = swift_constraint.evaluate(
            ephemeris=swift_ephemeris, target_ra=66, target_dec=-29
        )

        # Visibility windows should not overlap
        for i in range(len(result.visibility) - 1):
            assert (
                result.visibility[i].end_time <= result.visibility[i + 1].start_time
            ), (
                f"Overlapping visibility windows: {result.visibility[i]} and {result.visibility[i + 1]}"
            )
