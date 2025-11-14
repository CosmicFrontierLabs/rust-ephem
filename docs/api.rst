Python API
==========

This page documents the main Python API exported by the `rust_ephem` extension
module. The native extension is built with `maturin` and exposed under the
module name ``rust_ephem``.

Module Reference
----------------

.. automodule:: rust_ephem
  :members:
  :undoc-members:
  :show-inheritance:
  :no-index:

API Overview
------------

The module exposes the following primary classes and helper functions. If the
compiled extension is not available at documentation build time these names
may be mocked (see `docs/README.md`).

Classes
^^^^^^^

**TLEEphemeris**
  Propagate Two-Line Element (TLE) sets with SGP4 and convert to coordinate frames.
  
  **Constructor:**
    ``TLEEphemeris(tle1, tle2, begin, end, step_size=60, *, polar_motion=False)``
  
  **Attributes (read-only):**
    * ``teme_pv`` — Position/velocity in TEME frame (PositionVelocityData)
    * ``itrs_pv`` — Position/velocity in ITRS frame (PositionVelocityData)
    * ``gcrs_pv`` — Position/velocity in GCRS frame (PositionVelocityData)
    * ``sun_pv`` — Sun position/velocity in GCRS frame (PositionVelocityData)
    * ``moon_pv`` — Moon position/velocity in GCRS frame (PositionVelocityData)
    * ``timestamp`` — List of Python datetime objects
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``earth`` — Earth position as astropy SkyCoord
    * ``sun`` — Sun position as astropy SkyCoord
    * ``moon`` — Moon position as astropy SkyCoord
    * ``obsgeoloc`` — Observer geocentric location (alias for GCRS position)
    * ``obsgeovel`` — Observer geocentric velocity (alias for GCRS velocity)

**SPICEEphemeris**
  Access planetary ephemerides (SPK files) for celestial body positions.
  
  **Constructor:**
    ``SPICEEphemeris(spk_path, naif_id, begin, end, step_size=60, center_id=399, *, polar_motion=False)``
  
  **Attributes (read-only):**
    * ``gcrs_pv`` — Position/velocity in GCRS frame (PositionVelocityData)
    * ``itrs_pv`` — Position/velocity in ITRS frame (PositionVelocityData)
    * ``sun_pv`` — Sun position/velocity in GCRS frame (PositionVelocityData)
    * ``moon_pv`` — Moon position/velocity in GCRS frame (PositionVelocityData)
    * ``timestamp`` — List of Python datetime objects
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``earth`` — Earth position as astropy SkyCoord
    * ``sun`` — Sun position as astropy SkyCoord
    * ``moon`` — Moon position as astropy SkyCoord
    * ``obsgeoloc`` — Observer geocentric location (alias for GCRS position)
    * ``obsgeovel`` — Observer geocentric velocity (alias for GCRS velocity)

**GroundEphemeris**
  Ground-based observatory ephemeris for a fixed point on Earth's surface.
  
  **Constructor:**
    ``GroundEphemeris(latitude, longitude, height, begin, end, step_size=60, *, polar_motion=False)``
    
    * ``latitude`` — Geodetic latitude in degrees (-90 to 90)
    * ``longitude`` — Geodetic longitude in degrees (-180 to 180)
    * ``height`` — Altitude in meters above WGS84 ellipsoid
  
  **Attributes (read-only):**
    * ``latitude`` — Observatory latitude in degrees
    * ``longitude`` — Observatory longitude in degrees
    * ``height`` — Observatory height in meters
    * ``gcrs_pv`` — Position/velocity in GCRS frame (PositionVelocityData)
    * ``itrs_pv`` — Position/velocity in ITRS frame (PositionVelocityData)
    * ``sun_pv`` — Sun position/velocity in GCRS frame (PositionVelocityData)
    * ``moon_pv`` — Moon position/velocity in GCRS frame (PositionVelocityData)
    * ``timestamp`` — List of Python datetime objects
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``earth`` — Earth position as astropy SkyCoord
    * ``sun`` — Sun position as astropy SkyCoord
    * ``moon`` — Moon position as astropy SkyCoord
    * ``obsgeoloc`` — Observer geocentric location (alias for GCRS position)
    * ``obsgeovel`` — Observer geocentric velocity (alias for GCRS velocity)

**Constraint**
  Evaluate astronomical observation constraints against ephemeris data.
  
  **Constructor:**
    ``Constraint.from_json(json_config)`` — Create from JSON configuration string
  
  **Methods:**
    * ``evaluate(ephemeris, target_ra, target_dec)`` — Evaluate constraint against ephemeris data. Returns ``ConstraintResult``.

**ConstraintResult**
  Result of constraint evaluation containing violation information.
  
  **Attributes (read-only):**
    * ``violations`` — List of ``ConstraintViolation`` objects
    * ``all_satisfied`` — Boolean indicating if constraint was satisfied for entire time range
    * ``constraint_name`` — String name/description of the constraint
    * ``times`` — List of RFC3339 timestamp strings for evaluation times
  
  **Methods:**
    * ``total_violation_duration()`` — Get total duration of violations in seconds

**ConstraintViolation**
  Information about a specific constraint violation time window.
  
  **Attributes (read-only):**
    * ``start_time`` — Start time of violation window (ISO 8601 string)
    * ``end_time`` — End time of violation window (ISO 8601 string)
    * ``max_severity`` — Maximum severity of violation (0.0 = just violated, 1.0+ = severe)
    * ``description`` — Human-readable description of the violation

**PositionVelocityData**
  Container for position and velocity data returned by ephemeris calculations.
  
  **Attributes (read-only):**
    * ``position`` — NumPy array of positions (N × 3), in km
    * ``velocity`` — NumPy array of velocities (N × 3), in km/s
    * ``position_unit`` — String "km"
    * ``velocity_unit`` — String "km/s"

Functions
^^^^^^^^^

**Planetary Ephemeris Management**

* ``init_planetary_ephemeris(py_path)`` — Initialize an already-downloaded planetary SPK file.
* ``download_planetary_ephemeris(url, dest)`` — Download a planetary SPK file from a URL.
* ``ensure_planetary_ephemeris(py_path=None, download_if_missing=True, spk_url=None)`` — Download (if missing) and initialize planetary SPK lazily. Uses default de440s.bsp if no path provided.
* ``is_planetary_ephemeris_initialized()`` — Check if planetary ephemeris is initialized. Returns ``bool``.

**Time System Conversions**

* ``get_tai_utc_offset(py_datetime)`` — Get TAI-UTC offset (leap seconds) for a given datetime. Returns ``Optional[float]`` (seconds).
* ``get_ut1_utc_offset(py_datetime)`` — Get UT1-UTC offset for a given datetime. Returns ``float`` (seconds).
* ``is_ut1_available()`` — Check if UT1 data is available. Returns ``bool``.
* ``init_ut1_provider()`` — Initialize UT1 provider. Returns ``bool`` indicating success.

**Earth Orientation Parameters (EOP)**

* ``get_polar_motion(py_datetime)`` — Get polar motion parameters (x_p, y_p) for a given datetime. Returns ``Tuple[float, float]`` (arcseconds).
* ``is_eop_available()`` — Check if EOP data is available. Returns ``bool``.
* ``init_eop_provider()`` — Initialize EOP provider. Returns ``bool`` indicating success.

**Cache Management**

* ``get_cache_dir()`` — Get the path to the cache directory used by rust_ephem. Returns ``str``.

Constraint Configuration Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following Pydantic models are used to configure constraints. These can be serialized to/from JSON and support logical combinations using Python operators.

**SunConstraintConfig**
  Configuration for Sun proximity constraints.
  
  **Constructor:**
    ``SunConstraintConfig(min_angle=45.0)``
  
  **Attributes:**
    * ``type`` — Always "sun"
    * ``min_angle`` — Minimum angular separation from Sun in degrees (0-180)

**MoonConstraintConfig**
  Configuration for Moon proximity constraints.
  
  **Constructor:**
    ``MoonConstraintConfig(min_angle=30.0)``
  
  **Attributes:**
    * ``type`` — Always "moon"
    * ``min_angle`` — Minimum angular separation from Moon in degrees (0-180)

**EarthLimbConstraintConfig**
  Configuration for Earth limb proximity constraints.
  
  **Constructor:**
    ``EarthLimbConstraintConfig(min_angle=10.0)``
  
  **Attributes:**
    * ``type`` — Always "earth_limb"
    * ``min_angle`` — Minimum angular separation from Earth's limb in degrees (0-180)

**BodyConstraintConfig**
  Configuration for solar system body proximity constraints.
  
  **Constructor:**
    ``BodyConstraintConfig(body="Mars", min_angle=15.0)``
  
  **Attributes:**
    * ``type`` — Always "body"
    * ``body`` — Name of the solar system body (e.g., "Mars", "Jupiter")
    * ``min_angle`` — Minimum angular separation from body in degrees (0-180)

**EclipseConstraintConfig**
  Configuration for eclipse constraints (Earth shadow).
  
  **Constructor:**
    ``EclipseConstraintConfig(umbra_only=True)``
  
  **Attributes:**
    * ``type`` — Always "eclipse"
    * ``umbra_only`` — If True, only umbra counts. If False, includes penumbra.

**AndConstraintConfig**
  Logical AND combination of constraints.
  
  **Constructor:**
    ``AndConstraintConfig(constraints=[constraint1, constraint2])``
  
  **Attributes:**
    * ``type`` — Always "and"
    * ``constraints`` — List of constraint configurations to combine with AND

**OrConstraintConfig**
  Logical OR combination of constraints.
  
  **Constructor:**
    ``OrConstraintConfig(constraints=[constraint1, constraint2])``
  
  **Attributes:**
    * ``type`` — Always "or"
    * ``constraints`` — List of constraint configurations to combine with OR

**NotConstraintConfig**
  Logical NOT (negation) of a constraint.
  
  **Constructor:**
    ``NotConstraintConfig(constraint=some_constraint)``
  
  **Attributes:**
    * ``type`` — Always "not"
    * ``constraint`` — Constraint configuration to negate

**Constraint Operators**

Constraint configurations support Python bitwise operators for convenient combination:

* ``constraint1 & constraint2`` — Logical AND (equivalent to ``AndConstraintConfig``)
* ``constraint1 | constraint2`` — Logical OR (equivalent to ``OrConstraintConfig``)
* ``~constraint`` — Logical NOT (equivalent to ``NotConstraintConfig``)

Usage examples are provided in the examples section of the docs.
