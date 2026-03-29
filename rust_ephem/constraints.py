"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

import rust_ephem

from .ephemeris import Ephemeris

#: Default number of roll-angle samples used when sweeping spacecraft roll in
#: :meth:`~RustConstraintMixin.instantaneous_field_of_regard` when ``target_roll``
#: is not specified.  72 samples gives 5° resolution.
DEFAULT_N_ROLL_SAMPLES: int = 72

#: Default number of Fibonacci-sphere sky samples used by
#: :meth:`~RustConstraintMixin.instantaneous_field_of_regard`.
DEFAULT_N_POINTS: int = 20_000


class ConstraintViolation(BaseModel):
    """A time window where a constraint was violated."""

    start_time: datetime = Field(..., description="Start time of violation window")
    end_time: datetime = Field(..., description="End time of violation window")
    max_severity: float = Field(
        ..., description="Maximum severity of violation in this window"
    )
    description: str = Field(
        ..., description="Human-readable description of the violation"
    )


class ConstraintResult(BaseModel):
    """Result of constraint evaluation containing all violations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    violations: list[ConstraintViolation] = Field(
        default_factory=list, description="List of violation windows"
    )
    all_satisfied: bool = Field(
        ..., description="Whether constraint was satisfied for entire time range"
    )
    constraint_name: str = Field(..., description="Name/description of the constraint")

    # Store reference to Rust result for lazy access to timestamps/constraint_array
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._rust_result_ref = data.get("_rust_result_ref", None)
        # Populated when evaluate() sweeps all roll angles instead of a single Rust result.
        self._swept_timestamps: list[datetime] | None = data.get(
            "_swept_timestamps", None
        )
        self._swept_array: list[bool] | None = data.get("_swept_array", None)

    @property
    def timestamps(self) -> npt.NDArray[np.datetime64] | list[datetime]:
        """Evaluation timestamps (lazily accessed from Rust result)."""
        if self._swept_timestamps is not None:
            return self._swept_timestamps
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(
                npt.NDArray[np.datetime64] | list[datetime],
                self._rust_result_ref.timestamp,
            )
        return []

    @property
    def constraint_array(self) -> list[bool]:
        """
        Boolean array indicating constraint violations (lazily accessed from Rust result).

        Returns
        -------
        numpy.ndarray or list of bool
            Boolean array where True indicates the constraint is violated at that time,
            and False indicates the constraint is satisfied.
        """
        if self._swept_array is not None:
            return self._swept_array
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(list[bool], self._rust_result_ref.constraint_array)
        return []

    @property
    def visibility(self) -> list["rust_ephem.VisibilityWindow"]:
        """Visibility windows when the constraint is satisfied (target visible)."""
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(
                list["rust_ephem.VisibilityWindow"], self._rust_result_ref.visibility
            )
        return []

    def total_violation_duration(self) -> float:
        """Get the total duration of violations in seconds."""
        total_seconds = 0.0
        for violation in self.violations:
            total_seconds += (violation.end_time - violation.start_time).total_seconds()
        return total_seconds

    def in_constraint(self, time: datetime) -> bool:
        """Check if target is in-constraint at a given time.

        This method operates on timestamps from the evaluate() call.
        The given time must exist in the evaluated timestamps.

        Args:
            time: The datetime to check (must be in evaluated timestamps)

        Returns:
            True if the constraint is violated at this time (target is in-constraint),
            False if the constraint is satisfied (target is out-of-constraint).

        Raises:
            ValueError: If the time is not found in evaluated timestamps
        """
        if self._swept_timestamps is not None and self._swept_array is not None:
            try:
                idx = self._swept_timestamps.index(time)
                return self._swept_array[idx]
            except ValueError:
                raise ValueError(f"Time {time} not found in evaluated timestamps")
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(bool, self._rust_result_ref.in_constraint(time))
        raise ValueError(
            "ConstraintResult has no evaluated timestamps (was not created from evaluate())"
        )

    def __repr__(self) -> str:
        return f"ConstraintResult(constraint='{self.constraint_name}', violations={len(self.violations)}, all_satisfied={self.all_satisfied})"


if TYPE_CHECKING:
    pass


class RollReference(str, Enum):
    """Roll-zero reference axis for boresight offsets."""

    SUN = "sun"
    NORTH = "north"


class RustConstraintMixin(BaseModel):
    """Base class for Rust constraint configurations"""

    def _get_cached_rust_constraint(self) -> Any:
        """Return lazily cached Rust backend constraint for definition-only config."""
        if getattr(self, "_rust_constraint", None) is None:
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint

    def _resolve_rust_constraint(
        self,
        target_roll: float | None,
    ) -> Any:
        """Use cached backend unless evaluation-time roll is explicitly provided."""
        if target_roll is None:
            return self._get_cached_rust_constraint()
        return self._to_rust_constraint(
            target_roll=target_roll,
        )

    def _to_rust_constraint(
        self,
        target_roll: float | None = None,
        *,
        sweep_roll: bool = False,
    ) -> Any:
        """Build a Rust Constraint, injecting evaluation-time roll if needed.

        Args:
            target_roll: Spacecraft roll (degrees) added on top of each boresight
                instrument offset.  ``None`` means no spacecraft roll is applied.
            sweep_roll: When ``True`` (used internally by
                :meth:`instantaneous_field_of_regard`), boresight-offset nodes
                with non-zero pitch/yaw have their ``roll_deg`` set to ``null`` so
                the Rust layer sweeps all spacecraft rolls for FoR computation.
        """
        from rust_ephem import Constraint

        config = self.model_dump(mode="json")

        def apply_eval_roll(node: Any) -> None:
            if not isinstance(node, dict):
                return

            node_type = node.get("type")

            if node_type == "boresight_offset":
                pitch = float(node.get("pitch_deg", 0.0) or 0.0)
                yaw = float(node.get("yaw_deg", 0.0) or 0.0)
                has_offset = abs(pitch) > 1.0e-12 or abs(yaw) > 1.0e-12

                base_roll = float(node.get("roll_deg") or 0.0)
                base_clockwise = bool(node.get("roll_clockwise", False))
                base_reference = str(
                    node.get("roll_reference", "north") or "north"
                ).lower()
                if base_reference not in {"sun", "north"}:
                    raise ValueError("roll_reference must be either 'sun' or 'north'")
                base_ccw = -base_roll if base_clockwise else base_roll

                if sweep_roll and has_offset:
                    # Signal the Rust layer to sweep all spacecraft rolls for FoR.
                    node["roll_deg"] = None
                    node["roll_clockwise"] = base_clockwise
                    node["roll_reference"] = base_reference
                elif target_roll is not None:
                    # Combine instrument offset with spacecraft roll, both interpreted
                    # in the instrument's configured roll direction.
                    eval_ccw = (
                        -float(target_roll) if base_clockwise else float(target_roll)
                    )
                    total_ccw = base_ccw + eval_ccw
                    node["roll_deg"] = total_ccw
                    node["roll_clockwise"] = False
                    node["roll_reference"] = base_reference
                else:
                    # No spacecraft roll — preserve instrument offset unchanged.
                    node["roll_deg"] = base_roll
                    node["roll_clockwise"] = base_clockwise
                    node["roll_reference"] = base_reference

                inner = node.get("constraint")
                if isinstance(inner, dict):
                    apply_eval_roll(inner)
                return

            if node_type in {"and", "or", "xor", "at_least"}:
                for child in node.get("constraints", []):
                    apply_eval_roll(child)
                return

            if node_type == "not":
                apply_eval_roll(node.get("constraint"))

        apply_eval_roll(config)
        return Constraint.from_json(json.dumps(config))

    def _is_roll_dependent(self) -> bool:
        """Return True if this constraint tree has a boresight offset with non-zero pitch/yaw."""
        config = self.model_dump(mode="json")

        def check(node: Any) -> bool:
            if not isinstance(node, dict):
                return False
            node_type = node.get("type")
            if node_type == "boresight_offset":
                pitch = float(node.get("pitch_deg", 0.0) or 0.0)
                yaw = float(node.get("yaw_deg", 0.0) or 0.0)
                if abs(pitch) > 1.0e-12 or abs(yaw) > 1.0e-12:
                    return True
                return check(node.get("constraint"))
            if node_type in {"and", "or", "xor", "at_least"}:
                return any(check(c) for c in node.get("constraints", []))
            if node_type == "not":
                return bool(check(node.get("constraint")))
            return False

        return check(config)

    def evaluate(
        self,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
        target_roll: float | None = None,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
    ) -> ConstraintResult:
        """
        Evaluate the constraint using the Rust backend.

        This method lazily creates the corresponding Rust constraint
        object on first use.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate
            indices: Optional specific time index/indices to evaluate
            target_roll: Spacecraft roll angle (degrees).  When ``None`` (default) and
                the constraint has a boresight offset with non-zero pitch/yaw, sweeps
                ``n_roll_samples`` roll angles and returns violations only at
                timestamps where the target is blocked at **every** possible roll (i.e.
                no valid spacecraft orientation exists).  Pass an explicit roll value to
                evaluate at a fixed spacecraft roll.
            n_roll_samples: Number of roll angles to sweep when ``target_roll`` is ``None``
                and the constraint is roll-dependent.  Uniformly spaced over [0°, 360°).
                Default :data:`DEFAULT_N_ROLL_SAMPLES` (72 ≈ 5° resolution).

        Returns:
            ConstraintResult containing violation windows
        """
        if target_roll is None and self._is_roll_dependent():
            # Sweep all spacecraft roll angles; a timestamp is violated only if
            # blocked at every possible roll (no valid orientation exists).
            roll_step = 360.0 / n_roll_samples
            rust_results = [
                self._resolve_rust_constraint(target_roll=i * roll_step).evaluate(
                    ephemeris, target_ra, target_dec, times, indices
                )
                for i in range(n_roll_samples)
            ]
            arrays = [np.asarray(r.constraint_array, dtype=bool) for r in rust_results]
            combined: npt.NDArray[np.bool_] = arrays[0].copy()
            for arr in arrays[1:]:
                combined &= arr
            swept_timestamps = list(rust_results[0].timestamp)
            constraint_name = rust_results[0].constraint_name
            # Reconstruct violation windows from the AND'd boolean array.
            violations: list[ConstraintViolation] = []
            in_viol = False
            viol_start: datetime | None = None
            for dt, flag in zip(swept_timestamps, combined):
                if flag and not in_viol:
                    in_viol = True
                    viol_start = dt
                elif not flag and in_viol:
                    in_viol = False
                    violations.append(
                        ConstraintViolation(
                            start_time=cast(datetime, viol_start),
                            end_time=dt,
                            max_severity=1.0,
                            description=constraint_name,
                        )
                    )
            if in_viol and viol_start is not None and swept_timestamps:
                violations.append(
                    ConstraintViolation(
                        start_time=viol_start,
                        end_time=swept_timestamps[-1],
                        max_severity=1.0,
                        description=constraint_name,
                    )
                )
            return ConstraintResult(
                violations=violations,
                all_satisfied=not bool(combined.any()),
                constraint_name=constraint_name,
                _swept_timestamps=swept_timestamps,
                _swept_array=combined.tolist(),
            )

        rust_constraint = self._resolve_rust_constraint(
            target_roll=target_roll,
        )

        # Get the Rust result
        rust_result = rust_constraint.evaluate(
            ephemeris,
            target_ra,
            target_dec,
            times,
            indices,
        )

        # Convert to Pydantic model - Rust now returns datetime objects directly
        return ConstraintResult(
            violations=[
                ConstraintViolation(
                    start_time=v.start_time,
                    end_time=v.end_time,
                    max_severity=v.max_severity,
                    description=v.description,
                )
                for v in rust_result.violations
            ],
            all_satisfied=rust_result.all_satisfied,
            constraint_name=rust_result.constraint_name,
            _rust_result_ref=rust_result,
        )

    def in_constraint_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
        target_roll: float | None = None,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
    ) -> npt.NDArray[np.bool_]:
        """
        Check if targets are in-constraint for multiple RA/Dec positions (vectorized).

        This method lazily creates the corresponding Rust constraint
        object on first use and evaluates it for multiple RA/Dec positions.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ras: List of target right ascensions in degrees (ICRS/J2000)
            target_decs: List of target declinations in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate
            indices: Optional specific time index/indices to evaluate
            target_roll: Spacecraft roll angle (degrees).  When ``None`` (default) and
                the constraint has a boresight offset with non-zero pitch/yaw, sweeps
                ``n_roll_samples`` roll angles and returns ``True`` (violated)
                only if violated at **every** possible roll (i.e. no valid roll exists).
            n_roll_samples: Number of roll angles to sweep when ``target_roll`` is ``None``
                and the constraint is roll-dependent.  Uniformly spaced over [0°, 360°).
                Default :data:`DEFAULT_N_ROLL_SAMPLES` (72 ≈ 5° resolution).

        Returns:
            2D numpy array of shape (n_targets, n_times) with boolean violation status
        """
        if target_roll is None and self._is_roll_dependent():
            # Sweep all spacecraft roll angles; a cell is violated only if blocked at
            # every roll (AND across the sweep).
            roll_step = 360.0 / n_roll_samples
            combined: npt.NDArray[np.bool_] | None = None
            for i in range(n_roll_samples):
                r = i * roll_step
                arr = np.asarray(
                    self._resolve_rust_constraint(target_roll=r).in_constraint_batch(
                        ephemeris, target_ras, target_decs, times, indices
                    ),
                    dtype=bool,
                )
                combined = (
                    arr
                    if combined is None
                    else cast(npt.NDArray[np.bool_], combined & arr)
                )
            return cast(
                npt.NDArray[np.bool_],
                combined
                if combined is not None
                else np.ones((len(target_ras), 0), dtype=bool),
            )

        rust_constraint = self._resolve_rust_constraint(
            target_roll=target_roll,
        )
        return cast(
            npt.NDArray[np.bool_],
            rust_constraint.in_constraint_batch(
                ephemeris,
                target_ras,
                target_decs,
                times,
                indices,
            ),
        )

    def in_constraint(
        self,
        time: datetime | list[datetime] | npt.NDArray[np.datetime64],
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        target_roll: float | None = None,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
    ) -> bool | list[bool]:
        """Check if target is in-constraint at given time(s).

        This method performs full constraint evaluation for the given times.
        Use this to check constraint status without pre-computing evaluate().

        **API Note:** This differs from ConstraintResult.in_constraint() which
        operates on pre-evaluated timestamps. Use this method when you need
        to check arbitrary times, and use ConstraintResult.in_constraint()
        only for times already evaluated via evaluate().

        Args:
            time: The time(s) to check (must exist in ephemeris). Can be a single datetime,
                  list of datetimes, or numpy array of datetimes.
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)
            target_roll: Spacecraft roll angle (degrees).  When ``None`` (default) and
                the constraint has a boresight offset with non-zero pitch/yaw, sweeps
                ``n_roll_samples`` roll angles and returns ``True`` (violated)
                only if violated at **every** possible roll (i.e. no valid roll exists).
            n_roll_samples: Number of roll angles to sweep when ``target_roll`` is ``None``
                and the constraint is roll-dependent.  Uniformly spaced over [0°, 360°).
                Default :data:`DEFAULT_N_ROLL_SAMPLES` (72 ≈ 5° resolution).

        Returns:
            True if constraint is violated at the given time(s) (in-constraint).
            False if constraint is satisfied (out-of-constraint).
            Returns a single bool for a single time, or a list of bools for multiple times.
        """
        if target_roll is None and self._is_roll_dependent():
            # Sweep all spacecraft roll angles; violated only if blocked at every roll.
            roll_step = 360.0 / n_roll_samples
            parts: list[Any] = [
                self._resolve_rust_constraint(target_roll=i * roll_step).in_constraint(
                    time, ephemeris, target_ra, target_dec
                )
                for i in range(n_roll_samples)
            ]
            if isinstance(parts[0], bool):
                return all(parts)
            arr = np.array(parts, dtype=bool)  # (n_rolls, n_times)
            return list(arr.all(axis=0))

        rust_constraint = self._resolve_rust_constraint(
            target_roll=target_roll,
        )
        return cast(
            bool | list[bool],
            rust_constraint.in_constraint(
                time,
                ephemeris,
                target_ra,
                target_dec,
            ),
        )

    def roll_range(
        self,
        time: datetime,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
    ) -> list[tuple[float, float]]:
        """Return contiguous roll-angle intervals (degrees) where the constraint is satisfied.

        Sweeps ``n_roll_samples`` uniformly-spaced spacecraft roll angles over [0°, 360°),
        identifies those where the constraint is ``False`` (not violated), and collapses
        adjacent valid samples into ``(min_deg, max_deg)`` intervals.

        Args:
            time: A single datetime to evaluate (must exist in ephemeris).
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)
            n_roll_samples: Number of uniformly-spaced roll angles to test over [0°, 360°).
                Default 72 (5° resolution).

        Returns:
            List of ``(min_deg, max_deg)`` tuples, one per contiguous valid interval.
            Empty list if no roll is valid.
        """
        return self._get_cached_rust_constraint().roll_range(
            time, ephemeris, target_ra, target_dec, n_roll_samples
        )

    def instantaneous_field_of_regard(
        self,
        ephemeris: Ephemeris,
        time: datetime | None = None,
        index: int | None = None,
        n_points: int = DEFAULT_N_POINTS,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
        target_roll: float | None = None,
    ) -> float:
        """Compute instantaneous field of regard in steradians.

        Field of regard is the visible solid angle at a single timestamp,
        where visibility is defined by constraint not violated (False).

        When ``target_roll`` is not specified, boresight-offset constraints with
        non-zero pitch/yaw are evaluated by sweeping ``n_roll_samples`` spacecraft
        roll angles uniformly over [0°, 360°) and counting a sky direction as
        accessible if *any* roll satisfies the inner constraint.  This gives the
        maximum accessible sky over all possible spacecraft orientations.

        To evaluate at a specific spacecraft roll angle, pass ``target_roll``.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            time: Specific datetime to evaluate (must exist in ephemeris)
            index: Specific time index to evaluate
            n_points: Number of Fibonacci-sphere samples for sky integration
            n_roll_samples: Number of spacecraft roll angles to sweep when
                ``target_roll`` is not specified and boresight pitch/yaw offsets
                are present (uniformly spaced over [0°, 360°)).  Ignored when
                ``target_roll`` is given or no pitch/yaw offset is defined.
                Default :data:`DEFAULT_N_ROLL_SAMPLES` (5° resolution).
            target_roll: Spacecraft roll angle (degrees) about the boresight +X axis.
                When ``None`` (default), FoR sweeps all possible roll angles for
                boresight-offset constraints with non-zero pitch/yaw.

        Returns:
            Visible solid angle in steradians (range [0, 4π])

        Raises:
            ValueError: If exactly one of time/index is not provided
        """
        if target_roll is None:
            rust_constraint_any = cast(Any, self._to_rust_constraint(sweep_roll=True))
        else:
            rust_constraint_any = cast(
                Any, self._resolve_rust_constraint(target_roll=target_roll)
            )
        return float(
            rust_constraint_any.instantaneous_field_of_regard(
                ephemeris,
                time=time,
                index=index,
                n_points=n_points,
                n_roll_samples=n_roll_samples,
            )
        )

    def evaluate_moving_body(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float] | npt.ArrayLike | None = None,
        target_decs: list[float] | npt.ArrayLike | None = None,
        times: datetime | list[datetime] | None = None,
        body: str | int | None = None,
        use_horizons: bool = False,
        spice_kernel: str | None = None,
        target_roll: float | None = None,
    ) -> MovingVisibilityResult:
        """Evaluate constraint for a moving body (varying RA/Dec over time).

        This method evaluates the constraint for a body whose position changes over time,
        such as a comet, asteroid, or planet. It returns detailed results including
        per-timestamp violation status, visibility windows, and the body's coordinates.

        There are two ways to specify the body's position:
        1. Explicit coordinates: Provide `target_ras`, `target_decs`, and optionally `times`
        2. Body lookup: Provide `body` name/ID and optionally `use_horizons` to query positions

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ras: Array of right ascensions in degrees (ICRS/J2000)
            target_decs: Array of declinations in degrees (ICRS/J2000)
            times: Specific times to evaluate (must match ras/decs length)
            body: Body identifier (NAIF ID or name like "Jupiter", "90004910")
            use_horizons: If True, query JPL Horizons for body positions (default: False)
            spice_kernel: Path or URL to a SPICE kernel file for body positions.
                Can be a local path or URL (e.g., from JPL Horizons SPK files).

        Returns:
            MovingVisibilityResult with per-timestamp violation flags, visibility flags,
            RA/Dec coordinates, and merged visibility windows.

        Example:
            >>> # Using body name (queries SPICE or Horizons for positions)
            >>> result = constraint.evaluate_moving_body(ephem, body="Jupiter")
            >>> # Using explicit coordinates for a comet
            >>> result = constraint.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)
            >>> # Using a SPICE kernel for an asteroid
            >>> result = constraint.evaluate_moving_body(
            ...     ephem, body="2000001", spice_kernel="/path/to/ceres.bsp"
            ... )
        """
        rust_constraint = self._resolve_rust_constraint(
            target_roll=target_roll,
        )

        # Convert array-like to lists of floats if needed
        ras_list: list[float] | None = None
        decs_list: list[float] | None = None
        if target_ras is not None:
            ras_list = np.asarray(target_ras, dtype=float).tolist()
        if target_decs is not None:
            decs_list = np.asarray(target_decs, dtype=float).tolist()
        body_str = str(body) if body is not None else None

        # Call Rust implementation - returns MovingBodyResult object
        rust_result = rust_constraint.evaluate_moving_body(
            ephemeris,
            ras_list,
            decs_list,
            times,
            body_str,
            use_horizons,
            spice_kernel,
        )

        # Convert Rust VisibilityWindow objects to VisibilityWindowResult
        visibility_windows = [
            VisibilityWindowResult(
                start_time=w.start_time,
                end_time=w.end_time,
                duration_seconds=w.duration_seconds,
            )
            for w in rust_result.visibility
        ]

        # Convert constraint_array (violations) to visibility_flags (satisfied)
        visibility_flags = [not v for v in rust_result.constraint_array]

        return MovingVisibilityResult(
            timestamps=rust_result.timestamp,
            ras=rust_result.ras,
            decs=rust_result.decs,
            constraint_array=rust_result.constraint_array,
            visibility_flags=visibility_flags,
            visibility=visibility_windows,
            all_satisfied=rust_result.all_satisfied,
            constraint_name=rust_result.constraint_name,
        )

    def and_(self, other: ConstraintConfig) -> AndConstraint:
        """Combine this constraint with another using logical AND

        Args:
            other: Another constraint

        Returns:
            AndConstraint combining both constraints
        """
        return AndConstraint(constraints=[cast("ConstraintConfig", self), other])

    def or_(self, other: ConstraintConfig) -> OrConstraint:
        """Combine this constraint with another using logical OR

        Args:
            other: Another constraint

        Returns:
            OrConstraint combining both constraints
        """
        return OrConstraint(constraints=[cast("ConstraintConfig", self), other])

    def xor_(self, other: ConstraintConfig) -> XorConstraint:
        """Combine this constraint with another using logical XOR

        Args:
            other: Another constraint

        Returns:
            XorConstraint combining both constraints (violation when exactly one is violated)
        """
        return XorConstraint(constraints=[cast("ConstraintConfig", self), other])

    def at_least(
        self, min_violated: int, *others: ConstraintConfig
    ) -> AtLeastConstraint:
        """Combine this constraint with others using k-of-n violation logic.

        Args:
            min_violated: Minimum number of violated sub-constraints required
                to mark the combined constraint as violated.
            *others: Additional constraints to include with this constraint.

        Returns:
            AtLeastConstraint combining all constraints with threshold logic.
        """
        constraints = [cast("ConstraintConfig", self), *others]
        return AtLeastConstraint(
            min_violated=min_violated,
            constraints=constraints,
        )

    def not_(self) -> NotConstraint:
        """Negate this constraint using logical NOT

        Returns:
            NotConstraint negating this constraint
        """
        return NotConstraint(constraint=cast("ConstraintConfig", self))

    def __and__(self, other: ConstraintConfig) -> AndConstraint:
        """Combine constraints using & operator (logical AND)

        Args:
            other: Another constraint

        Returns:
            AndConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> combined = sun & moon
        """
        return self.and_(other)

    def __or__(self, other: ConstraintConfig) -> OrConstraint:
        """Combine constraints using | operator (logical OR)

        Args:
            other: Another constraint

        Returns:
            OrConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> combined = sun | moon
        """
        return self.or_(other)

    def __xor__(self, other: ConstraintConfig) -> XorConstraint:
        """Combine constraints using ^ operator (logical XOR)

        Args:
            other: Another constraint

        Returns:
            XorConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> exclusive = sun ^ moon
        """
        return self.xor_(other)

    def __invert__(self) -> NotConstraint:
        """Negate constraint using ~ operator (logical NOT)

        Returns:
            NotConstraint negating this constraint

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> not_sun = ~sun
        """
        return self.not_()

    def boresight_offset(
        self,
        roll_deg: float = 0.0,
        roll_clockwise: bool = False,
        roll_reference: RollReference = RollReference.NORTH,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> BoresightOffsetConstraint:
        """Wrap this constraint with a fixed boresight Euler-angle offset.

        This is useful for shared-axis multi-instrument systems where a secondary
        instrument is physically offset from the primary pointing direction.

        Args:
            roll_deg: Fixed instrument roll offset about the boresight +X axis,
                in degrees, relative to the spacecraft's nominal roll frame.
                Default 0.0 (instrument aligned with spacecraft frame).
            roll_clockwise: If True, positive roll is clockwise looking along +X.
            roll_reference: Roll-zero reference axis. ``"north"`` (default) sets
                roll=0 where +Z is celestial-north projected in the boresight-
                normal plane. ``"sun"`` sets roll=0 where +Z is Sun-projected in
                that plane.
            pitch_deg: Pitch angle about +Y in degrees
            yaw_deg: Yaw angle about +Z in degrees

        Returns:
            BoresightOffsetConstraint wrapping this constraint
        """
        return BoresightOffsetConstraint(
            constraint=cast("ConstraintConfig", self),
            roll_deg=roll_deg,
            roll_clockwise=roll_clockwise,
            roll_reference=roll_reference,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )


class SunConstraint(RustConstraintMixin):
    """Sun proximity constraint

    Ensures target maintains minimum angular separation from Sun.

    Attributes:
        type: Always "sun"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["sun"] = "sun"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Sun in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from Sun in degrees"
    )


class EarthLimbConstraint(RustConstraintMixin):
    """Earth limb avoidance constraint

    Ensures target maintains minimum angular separation from Earth's limb.
    For ground observers, optionally accounts for geometric horizon dip and atmospheric refraction.

    Attributes:
        type: Always "earth_limb"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
        include_refraction: Include atmospheric refraction correction (~0.57°) for ground observers (default: False)
        horizon_dip: Include geometric horizon dip correction for ground observers (default: False)
    """

    type: Literal["earth_limb"] = "earth_limb"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Earth's limb in degrees"
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from Earth's limb in degrees",
    )
    include_refraction: bool = Field(
        default=False,
        description="Include atmospheric refraction correction for ground observers",
    )
    horizon_dip: bool = Field(
        default=False,
        description="Include geometric horizon dip correction for ground observers",
    )


class BodyConstraint(RustConstraintMixin):
    """Solar system body proximity constraint

    Ensures target maintains minimum angular separation from specified body.

    Attributes:
        type: Always "body"
        body: Name of the solar system body (e.g., "Mars", "Jupiter")
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["body"] = "body"
    body: str = Field(..., description="Name of the solar system body")
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from body in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from body in degrees"
    )


class MoonConstraint(RustConstraintMixin):
    """Moon proximity constraint

    Ensures target maintains minimum angular separation from Moon.

    Attributes:
        type: Always "moon"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["moon"] = "moon"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Moon in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from Moon in degrees"
    )


class EclipseConstraint(RustConstraintMixin):
    """Eclipse constraint

    Checks if observer is in Earth's shadow (umbra and/or penumbra).
    Assumes Earth-centered ephemerides (Earth at the origin); results are
    undefined for other centers.

    Attributes:
        type: Always "eclipse"
        umbra_only: If True, only umbra counts. If False, includes penumbra.
    """

    type: Literal["eclipse"] = "eclipse"
    umbra_only: bool = Field(
        default=True, description="Count only umbra (True) or include penumbra (False)"
    )


class AndConstraint(RustConstraintMixin):
    """Logical AND constraint combinator

    Satisfied only if ALL sub-constraints are satisfied.

    Attributes:
        type: Always "and"
        constraints: List of constraints to combine with AND
    """

    type: Literal["and"] = "and"
    constraints: list[ConstraintConfig] = Field(
        ..., min_length=1, description="Constraints to AND together"
    )


class OrConstraint(RustConstraintMixin):
    """Logical OR constraint combinator

    Satisfied if ANY sub-constraint is satisfied.

    Attributes:
        type: Always "or"
        constraints: List of constraints to combine with OR
    """

    type: Literal["or"] = "or"
    constraints: list[ConstraintConfig] = Field(
        ..., min_length=1, description="Constraints to OR together"
    )


class XorConstraint(RustConstraintMixin):
    """Logical XOR constraint combinator

    Satisfied if EXACTLY ONE sub-constraint is satisfied.

    Attributes:
        type: Always "xor"
        constraints: List of constraints to combine with XOR (minimum 2)
    """

    type: Literal["xor"] = "xor"
    constraints: list[ConstraintConfig] = Field(
        ...,
        min_length=2,
        description="Constraints to XOR together (exactly one satisfied)",
    )


class AtLeastConstraint(RustConstraintMixin):
    """Threshold combinator (k-of-n violation logic).

    The combined constraint is violated when at least ``min_violated`` of
    the sub-constraints are violated.

    Attributes:
        type: Always "at_least"
        min_violated: Threshold k (must be >= 1 and <= number of constraints)
        constraints: List of constraints to evaluate
    """

    type: Literal["at_least"] = "at_least"
    min_violated: int = Field(
        ..., ge=1, description="Minimum violated sub-constraints required (k)"
    )
    constraints: list[ConstraintConfig] = Field(
        ..., min_length=1, description="Constraints in threshold combinator"
    )

    @model_validator(mode="after")
    def validate_min_violated(self) -> AtLeastConstraint:
        if self.min_violated > len(self.constraints):
            raise ValueError(
                "min_violated cannot exceed number of constraints "
                f"({self.min_violated} > {len(self.constraints)})"
            )
        return self


class NotConstraint(RustConstraintMixin):
    """Logical NOT constraint combinator

    Inverts a constraint - satisfied when inner constraint is violated.

    Attributes:
        type: Always "not"
        constraint: Constraint to negate
    """

    type: Literal["not"] = "not"
    constraint: ConstraintConfig = Field(..., description="Constraint to negate")


class BoresightOffsetConstraint(RustConstraintMixin):
    """Boresight offset wrapper for shared-axis multi-instrument constraints.

    Wraps another constraint and evaluates it at a direction rotated by fixed
    Euler angles from the primary pointing direction.

    Attributes:
        type: Always "boresight_offset"
        constraint: Inner constraint evaluated at offset direction
        roll_deg: Fixed instrument roll offset about the boresight +X axis in
            degrees, relative to the spacecraft's nominal roll frame.  Represents
            the physical mounting angle of the instrument.  Default ``0.0``
            (instrument aligned with spacecraft frame).
        roll_clockwise: If True, positive roll is clockwise looking along +X.
        roll_reference: Roll-zero reference axis ("sun" or "north")
        pitch_deg: Pitch angle about +Y in degrees
        yaw_deg: Yaw angle about +Z in degrees
    """

    type: Literal["boresight_offset"] = "boresight_offset"
    constraint: ConstraintConfig = Field(
        ..., description="Inner constraint evaluated at boresight-offset direction"
    )
    roll_deg: float = Field(
        default=0.0,
        description="Fixed instrument roll offset about boresight +X in degrees relative to spacecraft frame",
    )
    roll_clockwise: bool = Field(
        default=False,
        description="Interpret positive fixed boresight roll as clockwise looking along +X",
    )
    roll_reference: RollReference = Field(
        default=RollReference.NORTH,
        description="Roll-zero reference axis: 'sun' or 'north'",
    )
    pitch_deg: float = Field(default=0.0, description="Pitch angle about +Y in degrees")
    yaw_deg: float = Field(default=0.0, description="Yaw angle about +Z in degrees")


class DaytimeConstraint(RustConstraintMixin):
    """Daytime visibility constraint

    Prevents observations during daytime hours,
    with configurable twilight definitions.

    Attributes:
        type: Always "daytime"
        twilight: Twilight definition ("civil", "nautical", "astronomical", "none")
    """

    type: Literal["daytime"] = "daytime"
    twilight: Literal["civil", "nautical", "astronomical", "none"] = Field(
        default="civil", description="Twilight definition for daytime boundary"
    )


class AirmassConstraint(RustConstraintMixin):
    """Airmass constraint

    Limits observations based on atmospheric airmass (secant of zenith angle).
    Lower airmass values indicate better observing conditions.

    Attributes:
        type: Always "airmass"
        min_airmass: Minimum allowed airmass (≥1.0), optional
        max_airmass: Maximum allowed airmass (>0.0)
    """

    type: Literal["airmass"] = "airmass"
    min_airmass: float | None = Field(
        default=None, ge=1.0, description="Minimum allowed airmass"
    )
    max_airmass: float = Field(..., ge=1.0, description="Maximum allowed airmass")

    @model_validator(mode="after")
    def validate_airmass_values(self) -> AirmassConstraint:
        if self.min_airmass is not None and self.max_airmass < self.min_airmass:
            raise ValueError("max_airmass must be >= min_airmass")
        return self


class MoonPhaseConstraint(RustConstraintMixin):
    """Moon phase constraint

    Limits observations based on Moon illumination fraction and distance.

    Attributes:
        type: Always "moon_phase"
        min_illumination: Minimum allowed illumination fraction (0.0-1.0), optional
        max_illumination: Maximum allowed illumination fraction (0.0-1.0)
        min_distance: Minimum allowed Moon distance in degrees from target, optional
        max_distance: Maximum allowed Moon distance in degrees from target, optional
        enforce_when_below_horizon: Whether to enforce constraint when Moon is below horizon
        moon_visibility: Moon visibility requirement ("full" or "partial")
    """

    type: Literal["moon_phase"] = "moon_phase"
    min_illumination: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum allowed illumination fraction",
    )
    max_illumination: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum allowed illumination fraction"
    )
    min_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum allowed Moon distance in degrees from target",
    )
    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed Moon distance in degrees from target",
    )
    enforce_when_below_horizon: bool = Field(
        default=False,
        description="Whether to enforce constraint when Moon is below horizon",
    )
    moon_visibility: Literal["full", "partial"] = Field(
        default="full",
        description="Moon visibility requirement: 'full' (only when fully above horizon) or 'partial' (when any part visible)",
    )

    @model_validator(mode="after")
    def validate_moon_phase_values(self) -> MoonPhaseConstraint:
        if (
            self.min_illumination is not None
            and self.max_illumination < self.min_illumination
        ):
            raise ValueError("max_illumination must be >= min_illumination")
        if (
            self.min_distance is not None
            and self.max_distance is not None
            and self.max_distance < self.min_distance
        ):
            raise ValueError("max_distance must be >= min_distance")
        return self


class SAAConstraint(RustConstraintMixin):
    """South Atlantic Anomaly constraint

    Limits observations based on whether the spacecraft is within a defined
    geographic region (typically the South Atlantic Anomaly).

    Attributes:
        type: Always "saa"
        polygon: List of (longitude, latitude) pairs defining the region boundary
    """

    type: Literal["saa"] = "saa"
    polygon: list[tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="List of (longitude, latitude) pairs defining the region boundary in degrees",
    )


class AltAzConstraint(RustConstraintMixin):
    """Altitude/Azimuth constraint

    Limits observations based on target's altitude and azimuth angles
    from the observer's location. Can use simple min/max ranges or a
    custom polygon defining an allowed region.

    Attributes:
        type: Always "alt_az"
        min_altitude: Minimum allowed altitude in degrees (0-90), optional
        max_altitude: Maximum allowed altitude in degrees (0-90), optional
        min_azimuth: Minimum allowed azimuth in degrees (0-360), optional
        max_azimuth: Maximum allowed azimuth in degrees (0-360), optional
        polygon: List of (altitude, azimuth) pairs defining allowed region, optional
    """

    type: Literal["alt_az"] = "alt_az"
    min_altitude: float | None = Field(
        default=None, ge=0.0, le=90.0, description="Minimum allowed altitude in degrees"
    )
    max_altitude: float | None = Field(
        default=None, ge=0.0, le=90.0, description="Maximum allowed altitude in degrees"
    )
    min_azimuth: float | None = Field(
        default=None, ge=0.0, le=360.0, description="Minimum allowed azimuth in degrees"
    )
    max_azimuth: float | None = Field(
        default=None, ge=0.0, le=360.0, description="Maximum allowed azimuth in degrees"
    )
    polygon: list[tuple[float, float]] | None = Field(
        default=None,
        description="List of (altitude, azimuth) pairs in degrees defining allowed region",
    )


class OrbitRamConstraint(RustConstraintMixin):
    """Orbit RAM direction constraint

    Ensures target maintains minimum angular separation from the spacecraft's
    velocity vector (RAM direction). Useful for avoiding pointing
    directions that may cause contamination.

    Attributes:
        type: Always "orbit_ram"
        min_angle: Minimum allowed angular separation from RAM direction in degrees (0-180)
        max_angle: Maximum allowed angular separation from RAM direction in degrees (0-180), optional
    """

    type: Literal["orbit_ram"] = "orbit_ram"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from RAM direction in degrees"
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from RAM direction in degrees",
    )


class OrbitPoleConstraint(RustConstraintMixin):
    """Orbit pole direction constraint

    Ensures target maintains minimum angular separation from both the north and south
    orbital poles (directions perpendicular to the orbital plane). Useful for maintaining
    specific orientations relative to the spacecraft's orbit.

    Attributes:
        type: Always "orbit_pole"
        min_angle: Minimum allowed angular separation from both orbital poles in degrees (0-180)
        max_angle: Maximum allowed angular separation from both orbital poles in degrees (0-180), optional
        earth_limb_pole: If True, pole avoidance angle is earth_radius_deg + min_angle - 90.
                        Used for NASA's Neil Gehrels Swift Observatory where the pole is an emergent
                        property of Earth size plus Earth limb avoidance angle > 90°.
    """

    type: Literal["orbit_pole"] = "orbit_pole"
    min_angle: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description="Minimum angle from both orbital poles in degrees",
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from both orbital poles in degrees",
    )
    earth_limb_pole: bool = Field(
        default=False,
        description="If True, pole avoidance angle is earth_radius_deg + min_angle - 90",
    )


# Union type for all constraints
ConstraintConfig = Union[
    SunConstraint,
    MoonConstraint,
    EclipseConstraint,
    EarthLimbConstraint,
    BodyConstraint,
    DaytimeConstraint,
    AirmassConstraint,
    MoonPhaseConstraint,
    OrbitRamConstraint,
    OrbitPoleConstraint,
    SAAConstraint,
    AltAzConstraint,
    AndConstraint,
    OrConstraint,
    XorConstraint,
    AtLeastConstraint,
    NotConstraint,
    BoresightOffsetConstraint,
]


# Update forward references after ConstraintConfig is defined
AndConstraint.model_rebuild()
OrConstraint.model_rebuild()
XorConstraint.model_rebuild()
AtLeastConstraint.model_rebuild()
NotConstraint.model_rebuild()
BoresightOffsetConstraint.model_rebuild()


# Type adapter for ConstraintConfig union
CombinedConstraintConfig: TypeAdapter[ConstraintConfig] = TypeAdapter(ConstraintConfig)


class VisibilityWindowResult(BaseModel):
    """Visibility window for a moving target."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float


class MovingVisibilityResult(BaseModel):
    """Result for moving target visibility evaluation."""

    timestamps: list[datetime]
    ras: list[float]  # Right ascension in degrees for each timestamp
    decs: list[float]  # Declination in degrees for each timestamp
    constraint_array: list[bool]
    visibility_flags: list[bool]
    visibility: list[VisibilityWindowResult]
    all_satisfied: bool
    constraint_name: str
