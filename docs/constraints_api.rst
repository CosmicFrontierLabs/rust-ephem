Constraints API Reference
=========================

This page provides complete API documentation for the constraint system in
``rust-ephem``. Constraints are used to evaluate observational restrictions
for satellite and astronomical observation planning.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

The constraint system provides two complementary APIs:

1. **Rust-backed Constraint class** — Low-level interface with factory methods
   for creating constraints directly in Rust. Faster for simple use cases.

2. **Pydantic configuration models** — Type-safe Python models that serialize
   to/from JSON and support operator-based composition. Recommended for most users.

Both APIs can be used interchangeably and produce identical results.

Quick Start
-----------

.. code-block:: python

   import rust_ephem
   from rust_ephem.constraints import SunConstraint, MoonConstraint
   from datetime import datetime, timezone

   # Ensure planetary ephemeris is loaded
   rust_ephem.ensure_planetary_ephemeris()

   # Create ephemeris
   ephem = rust_ephem.TLEEphemeris(
       norad_id=25544,  # ISS
       begin=datetime(2024, 1, 1, tzinfo=timezone.utc),
       end=datetime(2024, 1, 2, tzinfo=timezone.utc),
       step_size=300
   )

   # Create combined constraint using operators
   constraint = SunConstraint(min_angle=45.0) & MoonConstraint(min_angle=10.0)

   # Evaluate for a target (Crab Nebula)
   result = constraint.evaluate(ephem, target_ra=83.63, target_dec=22.01)

   print(f"All satisfied: {result.all_satisfied}")
   print(f"Violations: {len(result.violations)}")
   print(f"Visibility windows: {len(result.visibility)}")


Constraint Class (Rust Backend)
-------------------------------

The ``Constraint`` class provides the core constraint evaluation functionality
implemented in Rust for maximum performance.

Factory Methods
^^^^^^^^^^^^^^^

.. py:staticmethod:: Constraint.sun_proximity(min_angle, max_angle=None)

   Create a Sun proximity constraint.

   :param float min_angle: Minimum allowed angular separation from Sun in degrees (0-180)
   :param float max_angle: Maximum allowed angular separation from Sun in degrees (optional)
   :returns: A new Constraint instance
   :rtype: Constraint
   :raises ValueError: If angles are out of valid range

   **Example:**

   .. code-block:: python

      # Target must be at least 45° from Sun
      constraint = Constraint.sun_proximity(45.0)

      # Target must be between 30° and 120° from Sun
      constraint = Constraint.sun_proximity(30.0, 120.0)

.. py:staticmethod:: Constraint.moon_proximity(min_angle, max_angle=None)

   Create a Moon proximity constraint.

   :param float min_angle: Minimum allowed angular separation from Moon in degrees (0-180)
   :param float max_angle: Maximum allowed angular separation from Moon in degrees (optional)
   :returns: A new Constraint instance
   :rtype: Constraint
   :raises ValueError: If angles are out of valid range

   **Example:**

   .. code-block:: python

      # Target must be at least 10° from Moon
      constraint = Constraint.moon_proximity(10.0)

.. py:staticmethod:: Constraint.earth_limb(min_angle, max_angle=None)

   Create an Earth limb avoidance constraint.

   For spacecraft, this ensures the target is sufficiently above Earth's limb
   as seen from the spacecraft position.

   :param float min_angle: Additional margin beyond Earth's apparent angular radius (degrees)
   :param float max_angle: Maximum allowed angular separation from Earth limb (degrees, optional)
   :returns: A new Constraint instance
   :rtype: Constraint
   :raises ValueError: If angles are out of valid range

   **Example:**

   .. code-block:: python

      # Target must be at least 28° above Earth's limb
      constraint = Constraint.earth_limb(28.0)

.. py:staticmethod:: Constraint.body_proximity(body, min_angle, max_angle=None)

   Create a generic solar system body avoidance constraint.

   :param str body: Body identifier — NAIF ID or name (e.g., "Jupiter", "499", "Mars")
   :param float min_angle: Minimum allowed angular separation in degrees (0-180)
   :param float max_angle: Maximum allowed angular separation in degrees (optional)
   :returns: A new Constraint instance
   :rtype: Constraint
   :raises ValueError: If angles are out of valid range

   **Supported Bodies:**

   - Planet names: "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"
   - Planet barycenters: "Jupiter barycenter", "5" (NAIF ID)
   - Other bodies: "Pluto", various moons (depending on loaded kernels)

   .. note::
      Body availability depends on the ephemeris type and loaded SPICE kernels.
      The default ``de440s.bsp`` includes Sun, Moon, Earth, and planetary barycenters.

   **Example:**

   .. code-block:: python

      # Target must be at least 15° from Mars
      constraint = Constraint.body_proximity("Mars", 15.0)

      # Using NAIF ID (5 = Jupiter barycenter)
      constraint = Constraint.body_proximity("5", 20.0)

.. py:staticmethod:: Constraint.eclipse(umbra_only=True)

   Create an eclipse constraint that detects when the observer is in Earth's shadow.

   :param bool umbra_only: If True, only umbra counts as eclipse. If False, penumbra also counts.
   :returns: A new Constraint instance
   :rtype: Constraint

   **Example:**

   .. code-block:: python

      # Constraint violated only in umbra (full shadow)
      constraint = Constraint.eclipse(umbra_only=True)

      # Constraint violated in both umbra and penumbra
      constraint = Constraint.eclipse(umbra_only=False)

Logical Combinators
^^^^^^^^^^^^^^^^^^^

.. py:staticmethod:: Constraint.and_(*constraints)

   Combine constraints with logical AND.

   :param constraints: Variable number of Constraint objects
   :returns: A new Constraint that is satisfied only if ALL input constraints are satisfied
   :rtype: Constraint
   :raises ValueError: If no constraints provided

   **Example:**

   .. code-block:: python

      sun = Constraint.sun_proximity(45.0)
      moon = Constraint.moon_proximity(10.0)
      combined = Constraint.and_(sun, moon)

.. py:staticmethod:: Constraint.or_(*constraints)

   Combine constraints with logical OR.

   :param constraints: Variable number of Constraint objects
   :returns: A new Constraint that is satisfied if ANY input constraint is satisfied
   :rtype: Constraint
   :raises ValueError: If no constraints provided

   **Example:**

   .. code-block:: python

      eclipse = Constraint.eclipse()
      earth_limb = Constraint.earth_limb(20.0)
      either = Constraint.or_(eclipse, earth_limb)

.. py:staticmethod:: Constraint.xor_(*constraints)

   Combine constraints with logical XOR.

   :param constraints: Variable number of Constraint objects (minimum 2)
   :returns: A new Constraint that is violated when EXACTLY ONE input constraint is violated
   :rtype: Constraint
   :raises ValueError: If fewer than two constraints are provided

   **Violation Semantics:**

   - XOR is violated when exactly one sub-constraint is violated
   - XOR is satisfied when zero or more than one sub-constraints are violated

   **Example:**

   .. code-block:: python

      sun = Constraint.sun_proximity(45.0)
      moon = Constraint.moon_proximity(10.0)
      exclusive = Constraint.xor_(sun, moon)

.. py:staticmethod:: Constraint.not_(constraint)

   Negate a constraint with logical NOT.

   :param Constraint constraint: Constraint to negate
   :returns: A new Constraint that is satisfied when the input is violated
   :rtype: Constraint

   **Example:**

   .. code-block:: python

      eclipse = Constraint.eclipse()
      not_eclipse = Constraint.not_(eclipse)  # Satisfied when NOT in eclipse

.. py:staticmethod:: Constraint.from_json(json_str)

   Create a constraint from a JSON string configuration.

   :param str json_str: JSON representation of the constraint configuration
   :returns: A new Constraint instance
   :rtype: Constraint
   :raises ValueError: If JSON is invalid or contains unknown constraint type

   **JSON Format Examples:**

   Simple constraints:

   .. code-block:: json

      {"type": "sun", "min_angle": 45.0}

   .. code-block:: json

      {"type": "moon", "min_angle": 10.0, "max_angle": 90.0}

   .. code-block:: json

      {"type": "eclipse", "umbra_only": true}

   .. code-block:: json

      {"type": "earth_limb", "min_angle": 28.0}

   .. code-block:: json

      {"type": "body", "body": "Mars", "min_angle": 15.0}

   Logical combinators:

   .. code-block:: json

      {"type": "and", "constraints": [{"type": "sun", "min_angle": 45.0}, {"type": "moon", "min_angle": 10.0}]}

   .. code-block:: json

      {"type": "not", "constraint": {"type": "eclipse", "umbra_only": true}}

   **Example:**

   .. code-block:: python

      json_config = '{"type": "sun", "min_angle": 45.0}'
      constraint = Constraint.from_json(json_config)

Evaluation Methods
^^^^^^^^^^^^^^^^^^

.. py:method:: Constraint.evaluate(ephemeris, target_ra, target_dec, times=None, indices=None)

   Evaluate constraint against ephemeris data.

   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param float target_ra: Target right ascension in degrees (ICRS/J2000)
   :param float target_dec: Target declination in degrees (ICRS/J2000)
   :param times: Optional specific time(s) to evaluate (datetime or list of datetimes)
   :param indices: Optional specific time index/indices to evaluate (int or list of ints)
   :returns: ConstraintResult containing violation windows
   :rtype: ConstraintResult
   :raises ValueError: If both times and indices are provided, or if times/indices not found
   :raises TypeError: If ephemeris type is not supported

   **Example:**

   .. code-block:: python

      result = constraint.evaluate(ephem, target_ra=83.63, target_dec=22.01)

      # Evaluate at specific times
      from datetime import datetime, timezone
      times = [
          datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
          datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc),
      ]
      result = constraint.evaluate(ephem, 83.63, 22.01, times=times)

      # Evaluate at specific indices
      result = constraint.evaluate(ephem, 83.63, 22.01, indices=[0, 10, 20])

.. py:method:: Constraint.in_constraint_batch(ephemeris, target_ras, target_decs, times=None, indices=None)

   Check if targets are in-constraint for multiple RA/Dec positions (vectorized).

   This method is **3-50x faster** than calling ``evaluate()`` in a loop when
   you need to check many target positions.

   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param list target_ras: List of target right ascensions in degrees (ICRS/J2000)
   :param list target_decs: List of target declinations in degrees (ICRS/J2000)
   :param times: Optional specific time(s) to evaluate
   :param indices: Optional specific time index/indices to evaluate
   :returns: 2D numpy boolean array of shape (n_targets, n_times)
   :rtype: numpy.ndarray

   **Return Value:**

   The returned array has shape ``(n_targets, n_times)`` where:

   - ``violations[i, j] = True`` means target ``i`` **violates** the constraint at time ``j``
   - ``violations[i, j] = False`` means target ``i`` **satisfies** the constraint at time ``j``

   **Example:**

   .. code-block:: python

      import numpy as np

      # Check 1000 random targets
      target_ras = np.random.uniform(0, 360, 1000)
      target_decs = np.random.uniform(-90, 90, 1000)

      violations = constraint.in_constraint_batch(ephem, target_ras, target_decs)

      print(f"Shape: {violations.shape}")  # (1000, n_times)

      # Count violations per target
      violation_counts = violations.sum(axis=1)

      # Find targets that never violate
      always_visible = np.where(violation_counts == 0)[0]

.. py:method:: Constraint.in_constraint(time, ephemeris, target_ra, target_dec)

   Check if the target satisfies the constraint at given time(s).

   This method accepts single times, lists of times, or numpy arrays of times.
   For multiple times, it efficiently uses batch evaluation internally.

   :param time: The time(s) to check (must exist in ephemeris timestamps).
                Can be a single datetime, list of datetimes, or numpy array of datetimes.
   :type time: datetime or list[datetime] or numpy.ndarray
   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param float target_ra: Target right ascension in degrees (ICRS/J2000)
   :param float target_dec: Target declination in degrees (ICRS/J2000)
   :returns: True if constraint is satisfied at the given time(s).
             Returns a single bool for a single time, or a list of bools for multiple times.
   :rtype: bool or list[bool]
   :raises ValueError: If time is not found in ephemeris timestamps

   **Examples:**

   .. code-block:: python

      import numpy as np
      from datetime import datetime, timezone

      time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

      # Single time
      is_visible = constraint.in_constraint(time, ephem, 83.63, 22.01)
      # Returns: bool

      # Multiple times as list
      times = [time, time]
      results = constraint.in_constraint(times, ephem, 83.63, 22.01)
      # Returns: [bool, bool]

      # Multiple times as numpy array
      times_array = np.array([time, time, time])
      results = constraint.in_constraint(times_array, ephem, 83.63, 22.01)
      # Returns: [bool, bool, bool]

Serialization Methods
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: Constraint.to_json()

   Get constraint configuration as JSON string.

   :returns: JSON string representation of the constraint
   :rtype: str

.. py:method:: Constraint.to_dict()

   Get constraint configuration as Python dictionary.

   :returns: Dictionary representation of the constraint
   :rtype: dict


Pydantic Configuration Models
-----------------------------

The ``rust_ephem.constraints`` module provides Pydantic models for type-safe
constraint configuration. These models support:

- JSON serialization/deserialization
- Validation of parameter ranges
- Python operator overloading for composition
- IDE autocompletion and type checking

Import all constraint models:

.. code-block:: python

   from rust_ephem.constraints import (
       SunConstraint,
       MoonConstraint,
       EarthLimbConstraint,
       BodyConstraint,
       EclipseConstraint,
       AndConstraint,
       OrConstraint,
       XorConstraint,
       NotConstraint,
       ConstraintConfig,
   )

SunConstraint
^^^^^^^^^^^^^

Sun proximity constraint ensuring target maintains minimum angular separation from Sun.

.. py:class:: SunConstraint(min_angle, max_angle=None)

   :param float min_angle: Minimum allowed angular separation in degrees (0-180, required)
   :param float max_angle: Maximum allowed angular separation in degrees (0-180, optional)

   **Attributes:**

   - ``type`` — Always ``"sun"`` (Literal)
   - ``min_angle`` — Minimum angle from Sun in degrees
   - ``max_angle`` — Maximum angle from Sun in degrees (or None)

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import SunConstraint

      # Simple minimum angle
      sun = SunConstraint(min_angle=45.0)

      # With maximum angle (target must be between 30° and 120° from Sun)
      sun = SunConstraint(min_angle=30.0, max_angle=120.0)

MoonConstraint
^^^^^^^^^^^^^^

Moon proximity constraint ensuring target maintains minimum angular separation from Moon.

.. py:class:: MoonConstraint(min_angle, max_angle=None)

   :param float min_angle: Minimum allowed angular separation in degrees (0-180, required)
   :param float max_angle: Maximum allowed angular separation in degrees (0-180, optional)

   **Attributes:**

   - ``type`` — Always ``"moon"`` (Literal)
   - ``min_angle`` — Minimum angle from Moon in degrees
   - ``max_angle`` — Maximum angle from Moon in degrees (or None)

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import MoonConstraint

      moon = MoonConstraint(min_angle=10.0)

EarthLimbConstraint
^^^^^^^^^^^^^^^^^^^

Earth limb avoidance constraint ensuring target is above Earth's horizon/limb.

.. py:class:: EarthLimbConstraint(min_angle, max_angle=None, include_refraction=False, horizon_dip=False)

   :param float min_angle: Minimum angular separation from Earth's limb in degrees (0-180, required)
   :param float max_angle: Maximum angular separation from Earth's limb in degrees (0-180, optional)
   :param bool include_refraction: Include atmospheric refraction correction (~0.57°) for ground observers (default: False)
   :param bool horizon_dip: Include geometric horizon dip correction for ground observers (default: False)

   **Attributes:**

   - ``type`` — Always ``"earth_limb"`` (Literal)
   - ``min_angle`` — Minimum angle from Earth's limb in degrees
   - ``max_angle`` — Maximum angle from Earth's limb in degrees (or None)
   - ``include_refraction`` — Whether to include atmospheric refraction
   - ``horizon_dip`` — Whether to include geometric horizon dip

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import EarthLimbConstraint

      # For spacecraft: target must be 28° above Earth's limb
      earth_limb = EarthLimbConstraint(min_angle=28.0)

      # For ground observers: include atmospheric effects
      earth_limb = EarthLimbConstraint(
          min_angle=10.0,
          include_refraction=True,
          horizon_dip=True
      )

BodyConstraint
^^^^^^^^^^^^^^

Generic solar system body proximity constraint.

.. py:class:: BodyConstraint(body, min_angle, max_angle=None)

   :param str body: Name of the solar system body (e.g., "Mars", "Jupiter")
   :param float min_angle: Minimum allowed angular separation in degrees (0-180, required)
   :param float max_angle: Maximum allowed angular separation in degrees (0-180, optional)

   **Attributes:**

   - ``type`` — Always ``"body"`` (Literal)
   - ``body`` — Name of the solar system body
   - ``min_angle`` — Minimum angle from body in degrees
   - ``max_angle`` — Maximum angle from body in degrees (or None)

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import BodyConstraint

      # Avoid Mars
      mars = BodyConstraint(body="Mars", min_angle=15.0)

      # Avoid Jupiter barycenter
      jupiter = BodyConstraint(body="Jupiter barycenter", min_angle=20.0)

EclipseConstraint
^^^^^^^^^^^^^^^^^

Eclipse constraint detecting when observer is in Earth's shadow.

.. py:class:: EclipseConstraint(umbra_only=True)

   :param bool umbra_only: If True, only umbra counts as eclipse. If False, includes penumbra. (default: True)

   **Attributes:**

   - ``type`` — Always ``"eclipse"`` (Literal)
   - ``umbra_only`` — Whether only umbra counts as eclipse

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import EclipseConstraint

      # Only detect full shadow (umbra)
      eclipse = EclipseConstraint(umbra_only=True)

      # Detect both umbra and penumbra
      eclipse = EclipseConstraint(umbra_only=False)

AndConstraint
^^^^^^^^^^^^^

Logical AND combination — satisfied only if ALL sub-constraints are satisfied.

.. py:class:: AndConstraint(constraints)

   :param list constraints: List of ConstraintConfig objects to combine (minimum 1)

   **Attributes:**

   - ``type`` — Always ``"and"`` (Literal)
   - ``constraints`` — List of constraints to AND together

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import AndConstraint, SunConstraint, MoonConstraint

      combined = AndConstraint(constraints=[
          SunConstraint(min_angle=45.0),
          MoonConstraint(min_angle=10.0),
      ])

OrConstraint
^^^^^^^^^^^^

Logical OR combination — satisfied if ANY sub-constraint is satisfied.

.. py:class:: OrConstraint(constraints)

   :param list constraints: List of ConstraintConfig objects to combine (minimum 1)

   **Attributes:**

   - ``type`` — Always ``"or"`` (Literal)
   - ``constraints`` — List of constraints to OR together

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import OrConstraint, EclipseConstraint, EarthLimbConstraint

      either = OrConstraint(constraints=[
          EclipseConstraint(),
          EarthLimbConstraint(min_angle=20.0),
      ])

XorConstraint
^^^^^^^^^^^^^

Logical XOR combination — violated when EXACTLY ONE sub-constraint is violated.

.. py:class:: XorConstraint(constraints)

   :param list constraints: List of ConstraintConfig objects (minimum 2)

   **Violation Semantics:**

   - XOR is **violated** when exactly one sub-constraint is violated
   - XOR is **satisfied** when zero or more than one sub-constraints are violated

   **Attributes:**

   - ``type`` — Always ``"xor"`` (Literal)
   - ``constraints`` — List of constraints (minimum 2) evaluated with XOR semantics

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import XorConstraint, SunConstraint, MoonConstraint

      exclusive = XorConstraint(constraints=[
          SunConstraint(min_angle=45.0),
          MoonConstraint(min_angle=30.0),
      ])

NotConstraint
^^^^^^^^^^^^^

Logical NOT — inverts a constraint (satisfied when inner constraint is violated).

.. py:class:: NotConstraint(constraint)

   :param constraint: ConstraintConfig object to negate

   **Attributes:**

   - ``type`` — Always ``"not"`` (Literal)
   - ``constraint`` — Constraint to negate

   **Example:**

   .. code-block:: python

      from rust_ephem.constraints import NotConstraint, EclipseConstraint

      # Satisfied when NOT in eclipse
      not_eclipse = NotConstraint(constraint=EclipseConstraint())


Operator Overloading
--------------------

All Pydantic constraint models support Python bitwise operators for intuitive
composition:

.. list-table:: Constraint Operators
   :header-rows: 1
   :widths: 20 30 50

   * - Operator
     - Equivalent
     - Description
   * - ``a & b``
     - ``AndConstraint([a, b])``
     - Logical AND — both must be satisfied
   * - ``a | b``
     - ``OrConstraint([a, b])``
     - Logical OR — at least one must be satisfied
   * - ``a ^ b``
     - ``XorConstraint([a, b])``
     - Logical XOR — violated when exactly one is violated
   * - ``~a``
     - ``NotConstraint(a)``
     - Logical NOT — inverts the constraint

**Example:**

.. code-block:: python

   from rust_ephem.constraints import (
       SunConstraint, MoonConstraint, EclipseConstraint, EarthLimbConstraint
   )

   # Build complex constraint with operators
   constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       ~EclipseConstraint(umbra_only=True)
   )

   # Equivalent to:
   # AndConstraint(constraints=[
   #     SunConstraint(min_angle=45.0),
   #     MoonConstraint(min_angle=10.0),
   #     NotConstraint(constraint=EclipseConstraint(umbra_only=True))
   # ])

   # Chain multiple operators
   complex_constraint = (
       (SunConstraint(min_angle=45.0) & MoonConstraint(min_angle=10.0)) |
       EarthLimbConstraint(min_angle=28.0)
   )


Common Methods (RustConstraintMixin)
------------------------------------

All Pydantic constraint models inherit these methods:

.. py:method:: evaluate(ephemeris, target_ra, target_dec, times=None, indices=None)

   Evaluate the constraint using the Rust backend.

   This method lazily creates the corresponding Rust constraint object on first use.

   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param float target_ra: Target right ascension in degrees (ICRS/J2000)
   :param float target_dec: Target declination in degrees (ICRS/J2000)
   :param times: Optional specific time(s) to evaluate
   :param indices: Optional specific time index/indices to evaluate
   :returns: ConstraintResult containing violation windows
   :rtype: ConstraintResult

.. py:method:: in_constraint_batch(ephemeris, target_ras, target_decs, times=None, indices=None)

   Check if targets are in-constraint for multiple RA/Dec positions (vectorized).

   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param list target_ras: List of target right ascensions in degrees
   :param list target_decs: List of target declinations in degrees
   :param times: Optional specific time(s) to evaluate
   :param indices: Optional specific time index/indices to evaluate
   :returns: 2D numpy array of shape (n_targets, n_times) with boolean violation status
   :rtype: numpy.ndarray

.. py:method:: in_constraint(time, ephemeris, target_ra, target_dec)

   Check if target violates the constraint at given time(s).

   :param time: The time(s) to check (must exist in ephemeris). Can be a single datetime,
                list of datetimes, or numpy array of datetimes.
   :type time: datetime or list[datetime] or numpy.ndarray
   :param ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
   :param float target_ra: Target right ascension in degrees
   :param float target_dec: Target declination in degrees
   :returns: True if constraint is satisfied at the given time(s). Returns a single bool
             for a single time, or a list of bools for multiple times.
   :rtype: bool or list[bool]

.. py:method:: and_(other)

   Combine this constraint with another using logical AND.

   :param other: Another ConstraintConfig
   :returns: AndConstraint combining both
   :rtype: AndConstraint

.. py:method:: or_(other)

   Combine this constraint with another using logical OR.

   :param other: Another ConstraintConfig
   :returns: OrConstraint combining both
   :rtype: OrConstraint

.. py:method:: xor_(other)

   Combine this constraint with another using logical XOR.

   :param other: Another ConstraintConfig
   :returns: XorConstraint combining both
   :rtype: XorConstraint

.. py:method:: not_()

   Negate this constraint using logical NOT.

   :returns: NotConstraint negating this constraint
   :rtype: NotConstraint


Result Classes
--------------

ConstraintResult
^^^^^^^^^^^^^^^^

Result of constraint evaluation containing all violation information.

.. py:class:: ConstraintResult

   **Attributes:**

   - ``violations`` (list[ConstraintViolation]) — List of violation time windows
   - ``all_satisfied`` (bool) — True if constraint was satisfied for entire time range
   - ``constraint_name`` (str) — Name/description of the constraint
   - ``timestamp`` (numpy.ndarray) — Array of datetime objects for each evaluation time (cached)
   - ``constraint_array`` (numpy.ndarray) — Boolean array where True = satisfied (cached)
   - ``visibility`` (list[VisibilityWindow]) — Contiguous windows when target is visible

   **Methods:**

   .. py:method:: total_violation_duration()

      Get the total duration of violations in seconds.

      :returns: Total violation duration in seconds
      :rtype: float

   .. py:method:: in_constraint(time)

      Check if the target is in-constraint at a given time.

      :param datetime time: A datetime object (must exist in evaluated timestamps)
      :returns: True if constraint is satisfied at the given time
      :rtype: bool
      :raises ValueError: If time is not in the evaluated timestamps

   **Example:**

   .. code-block:: python

      result = constraint.evaluate(ephem, 83.63, 22.01)

      print(f"Constraint: {result.constraint_name}")
      print(f"All satisfied: {result.all_satisfied}")
      print(f"Total violation duration: {result.total_violation_duration()} seconds")

      # Access visibility windows
      for window in result.visibility:
          print(f"Visible: {window.start_time} to {window.end_time}")

      # Efficient iteration using cached arrays
      for i, (time, satisfied) in enumerate(zip(result.timestamp, result.constraint_array)):
          if satisfied:
              print(f"Target visible at {time}")

ConstraintViolation
^^^^^^^^^^^^^^^^^^^

Information about a specific constraint violation time window.

.. py:class:: ConstraintViolation

   **Attributes:**

   - ``start_time`` (str) — Start time of violation window (ISO 8601 string)
   - ``end_time`` (str) — End time of violation window (ISO 8601 string)
   - ``max_severity`` (float) — Maximum severity of violation (0.0 = just violated, 1.0+ = severe)
   - ``description`` (str) — Human-readable description of the violation

   **Example:**

   .. code-block:: python

      for violation in result.violations:
          print(f"Violation: {violation.start_time} to {violation.end_time}")
          print(f"  Severity: {violation.max_severity:.2f}")
          print(f"  Description: {violation.description}")

VisibilityWindow
^^^^^^^^^^^^^^^^

Time window when the observation target is not constrained (visible).

.. py:class:: VisibilityWindow

   **Attributes:**

   - ``start_time`` (datetime) — Start time of visibility window
   - ``end_time`` (datetime) — End time of visibility window
   - ``duration_seconds`` (float) — Duration of the window in seconds (computed property)

   **Example:**

   .. code-block:: python

      for window in result.visibility:
          print(f"Window: {window.start_time} to {window.end_time}")
          print(f"  Duration: {window.duration_seconds / 3600:.2f} hours")


Type Aliases
------------

.. py:data:: ConstraintConfig

   Union type for all constraint configuration classes::

       ConstraintConfig = Union[
           SunConstraint,
           MoonConstraint,
           EclipseConstraint,
           EarthLimbConstraint,
           BodyConstraint,
           AndConstraint,
           OrConstraint,
           XorConstraint,
           NotConstraint,
       ]

   Use this type for function signatures that accept any constraint type.

.. py:data:: CombinedConstraintConfig

   Pydantic TypeAdapter for parsing constraint configurations from JSON::

       from rust_ephem.constraints import CombinedConstraintConfig

       json_str = '{"type": "sun", "min_angle": 45.0}'
       constraint = CombinedConstraintConfig.validate_json(json_str)


JSON Serialization
------------------

All Pydantic constraint models can be serialized to/from JSON:

.. code-block:: python

   from rust_ephem.constraints import SunConstraint, MoonConstraint

   # Create constraint
   constraint = SunConstraint(min_angle=45.0) & MoonConstraint(min_angle=10.0)

   # Serialize to JSON
   json_str = constraint.model_dump_json()
   # '{"type":"and","constraints":[{"type":"sun","min_angle":45.0,"max_angle":null},{"type":"moon","min_angle":10.0,"max_angle":null}]}'

   # Parse from JSON
   from rust_ephem.constraints import CombinedConstraintConfig
   parsed = CombinedConstraintConfig.validate_json(json_str)

   # Or use the Rust backend directly
   import rust_ephem
   rust_constraint = rust_ephem.Constraint.from_json(json_str)


Performance Guide
-----------------

The constraint system is optimized for high-performance evaluation. Follow these
guidelines for best performance:

Batch Evaluation (Fastest)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For evaluating many targets, use ``in_constraint_batch()``:

.. code-block:: python

   import numpy as np

   # Generate 10,000 target positions
   target_ras = np.random.uniform(0, 360, 10000)
   target_decs = np.random.uniform(-90, 90, 10000)

   # Single call evaluates all targets (3-50x faster than loop)
   violations = constraint.in_constraint_batch(ephem, target_ras, target_decs)

   # violations.shape = (10000, n_times)

**Performance by constraint type:**

- Sun/Moon proximity: ~3-4x speedup over loop
- Earth limb: ~5x speedup
- Eclipse: ~48x speedup (target-independent)
- Logical combinators: ~2-3x speedup

Single Target Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^

For a single target over many times:

.. code-block:: python

   # FAST: Evaluate once, use cached arrays
   result = constraint.evaluate(ephem, ra, dec)

   # Access cached arrays (90x faster on repeated access)
   times = result.timestamp
   satisfied = result.constraint_array

   # Find visibility windows directly
   visible_indices = np.where(satisfied)[0]

Single Time Check
^^^^^^^^^^^^^^^^^

For checking a single time:

.. code-block:: python

   # Use in_constraint() for single-time checks
   is_visible = constraint.in_constraint(time, ephem, ra, dec)

For checking multiple times efficiently:

.. code-block:: python

   # Use in_constraint() with arrays for multiple times
   times_array = ephem.timestamp[10:20]  # numpy array
   results = constraint.in_constraint(times_array, ephem, ra, dec)
   # Returns list of booleans

Anti-Patterns (Avoid)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # SLOW: Don't call in_constraint() in a loop
   for time in ephem.timestamp:
       if constraint.in_constraint(time, ephem, ra, dec):  # Re-evaluates every time!
           pass

   # SLOW: Don't call evaluate() for each target
   for ra, dec in zip(target_ras, target_decs):
       result = constraint.evaluate(ephem, ra, dec)  # Use in_constraint_batch() instead!

Subset Evaluation
^^^^^^^^^^^^^^^^^

Evaluate only specific times to reduce computation:

.. code-block:: python

   # Only evaluate first 10 and last 10 times
   indices = list(range(10)) + list(range(-10, 0))
   result = constraint.evaluate(ephem, ra, dec, indices=indices)

   # Only evaluate specific datetimes
   specific_times = [
       datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
       datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc),
   ]
   result = constraint.evaluate(ephem, ra, dec, times=specific_times)
