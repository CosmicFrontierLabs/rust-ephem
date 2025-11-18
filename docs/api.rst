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
    * ``sun_radius`` — Sun angular radius as astropy Quantity (degrees)
    * ``sun_radius_deg`` — Sun angular radius as NumPy array (degrees)
    * ``sun_radius_rad`` — Sun angular radius as NumPy array (radians)
    * ``moon_radius`` — Moon angular radius as astropy Quantity (degrees)
    * ``moon_radius_deg`` — Moon angular radius as NumPy array (degrees)
    * ``moon_radius_rad`` — Moon angular radius as NumPy array (radians)
    * ``earth_radius`` — Earth angular radius as astropy Quantity (degrees)
    * ``earth_radius_deg`` — Earth angular radius as NumPy array (degrees)
    * ``earth_radius_rad`` — Earth angular radius as NumPy array (radians)
  
  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime
      
      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

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
    * ``sun_radius`` — Sun angular radius as astropy Quantity (degrees)
    * ``sun_radius_deg`` — Sun angular radius as NumPy array (degrees)
    * ``sun_radius_rad`` — Sun angular radius as NumPy array (radians)
    * ``moon_radius`` — Moon angular radius as astropy Quantity (degrees)
    * ``moon_radius_deg`` — Moon angular radius as NumPy array (degrees)
    * ``moon_radius_rad`` — Moon angular radius as NumPy array (radians)
    * ``earth_radius`` — Earth angular radius as astropy Quantity (degrees)
    * ``earth_radius_deg`` — Earth angular radius as NumPy array (degrees)
    * ``earth_radius_rad`` — Earth angular radius as NumPy array (radians)
  
  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime
      
      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

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
    * ``sun_radius`` — Sun angular radius as astropy Quantity (degrees)
    * ``sun_radius_deg`` — Sun angular radius as NumPy array (degrees)
    * ``sun_radius_rad`` — Sun angular radius as NumPy array (radians)
    * ``moon_radius`` — Moon angular radius as astropy Quantity (degrees)
    * ``moon_radius_deg`` — Moon angular radius as NumPy array (degrees)
    * ``moon_radius_rad`` — Moon angular radius as NumPy array (radians)
    * ``earth_radius`` — Earth angular radius as astropy Quantity (degrees)
    * ``earth_radius_deg`` — Earth angular radius as NumPy array (degrees)
    * ``earth_radius_rad`` — Earth angular radius as NumPy array (radians)
  
  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime
      
      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``sun_position = eph.sun_pv.position[idx]``

**OEMEphemeris**
  Load and interpolate CCSDS Orbit Ephemeris Message (OEM) files for spacecraft ephemeris.
  
  The OEM file must use a GCRS-compatible reference frame such as J2000, EME2000, GCRF, or ICRF.
  Earth-fixed frames (e.g., ITRF) are not supported and will raise a ValueError.
  
  **Constructor:**
    ``OEMEphemeris(oem_file_path, begin, end, step_size=60, *, polar_motion=False)``
    
    * ``oem_file_path`` — Path to CCSDS OEM file (.oem)
    * ``begin`` — Start time for ephemeris (Python datetime)
    * ``end`` — End time for ephemeris (Python datetime)
    * ``step_size`` — Time step in seconds for interpolated ephemeris (default: 60)
    * ``polar_motion`` — Enable polar motion corrections (default: False)
  
  **Raises:**
    * ``ValueError`` — If reference frame is missing or incompatible with GCRS
  
  **Attributes (read-only):**
    * ``oem_pv`` — Original OEM state vectors (PositionVelocityData) without interpolation
    * ``oem_timestamp`` — Original OEM timestamps (list of datetime) without interpolation
    * ``gcrs_pv`` — Interpolated position/velocity in GCRS frame (PositionVelocityData)
    * ``itrs_pv`` — Position/velocity in ITRS frame (PositionVelocityData)
    * ``sun_pv`` — Sun position/velocity in GCRS frame (PositionVelocityData)
    * ``moon_pv`` — Moon position/velocity in GCRS frame (PositionVelocityData)
    * ``timestamp`` — List of Python datetime objects for interpolated ephemeris
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``earth`` — Earth position as astropy SkyCoord
    * ``sun`` — Sun position as astropy SkyCoord
    * ``moon`` — Moon position as astropy SkyCoord
    * ``obsgeoloc`` — Observer geocentric location (alias for GCRS position)
    * ``obsgeovel`` — Observer geocentric velocity (alias for GCRS velocity)
    * ``sun_radius`` — Sun angular radius as astropy Quantity (degrees)
    * ``sun_radius_deg`` — Sun angular radius as NumPy array (degrees)
    * ``sun_radius_rad`` — Sun angular radius as NumPy array (radians)
    * ``moon_radius`` — Moon angular radius as astropy Quantity (degrees)
    * ``moon_radius_deg`` — Moon angular radius as NumPy array (degrees)
    * ``moon_radius_rad`` — Moon angular radius as NumPy array (radians)
    * ``earth_radius`` — Earth angular radius as astropy Quantity (degrees)
    * ``earth_radius_deg`` — Earth angular radius as NumPy array (degrees)
    * ``earth_radius_rad`` — Earth angular radius as NumPy array (radians)
  
  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime
      
      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2032, 7, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

**Constraint**
  Evaluate astronomical observation constraints against ephemeris data.
  
  **Static Methods:**
    * ``Constraint.sun_proximity(min_angle, max_angle=None)`` — Create Sun proximity constraint
    * ``Constraint.moon_proximity(min_angle, max_angle=None)`` — Create Moon proximity constraint
    * ``Constraint.earth_limb(min_angle, max_angle=None)`` — Create Earth limb avoidance constraint
      * ``Constraint.earth_limb(min_angle, max_angle=None, include_refraction=False, horizon_dip=False)`` — Create Earth limb avoidance constraint
    * ``Constraint.body_proximity(body, min_angle, max_angle=None)`` — Create solar system body proximity constraint
    * ``Constraint.eclipse(umbra_only=True)`` — Create eclipse constraint
    * ``Constraint.and_(*constraints)`` — Combine constraints with logical AND
    * ``Constraint.or_(*constraints)`` — Combine constraints with logical OR
    * ``Constraint.xor_(*constraints)`` — Combine constraints with logical XOR (violation when exactly one sub-constraint is violated)
    * ``Constraint.not_(constraint)`` — Negate a constraint with logical NOT
    * ``Constraint.from_json(json_str)`` — Create constraint from JSON configuration
  
  **Methods:**
    * ``evaluate(ephemeris, target_ra, target_dec, times=None, indices=None)`` — Evaluate constraint against ephemeris data
      
      - ``ephemeris`` — TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris object
      - ``target_ra`` — Target right ascension in degrees (ICRS/J2000)
      - ``target_dec`` — Target declination in degrees (ICRS/J2000)
      - ``times`` — Optional: specific datetime(s) to evaluate (must exist in ephemeris)
      - ``indices`` — Optional: specific time index/indices to evaluate
      - Returns: ``ConstraintResult`` object
    
    * ``in_constraint(time, ephemeris, target_ra, target_dec)`` — Check if target is in-constraint at a single time
      
      - ``time`` — Python datetime object (must exist in ephemeris timestamps)
      - Returns: ``bool`` (True if constraint is satisfied, False if violated)
    
    * ``to_json()`` — Get constraint configuration as JSON string
    * ``to_dict()`` — Get constraint configuration as Python dictionary

**ConstraintResult**
  Result of constraint evaluation containing violation information.
  
  **Attributes (read-only):**
    * ``violations`` — List of ``ConstraintViolation`` objects
    * ``all_satisfied`` — Boolean indicating if constraint was satisfied for entire time range
    * ``constraint_name`` — String name/description of the constraint
    * ``timestamp`` — NumPy array of Python datetime objects (optimized with caching)
    * ``constraint_array`` — NumPy boolean array where True means constraint satisfied (optimized with caching)
    * ``visibility`` — List of ``VisibilityWindow`` objects for contiguous satisfied periods
  
  **Methods:**
    * ``total_violation_duration()`` — Get total duration of violations in seconds
    * ``in_constraint(time)`` — Check if constraint is satisfied at a given time
      
      - ``time`` — Python datetime object (must exist in result timestamps)
      - Returns: ``bool`` (True if satisfied, False if violated)

**ConstraintViolation**
  Information about a specific constraint violation time window.
  
  **Attributes (read-only):**
    * ``start_time`` — Start time of violation window (ISO 8601 string)
    * ``end_time`` — End time of violation window (ISO 8601 string)
    * ``max_severity`` — Maximum severity of violation (0.0 = just violated, 1.0+ = severe)
    * ``description`` — Human-readable description of the violation

**VisibilityWindow**
  Time window when observation target is not constrained (visible).
  
  **Attributes (read-only):**
    * ``start_time`` — Start time of visibility window (Python datetime)
    * ``end_time`` — End time of visibility window (Python datetime)
    * ``duration_seconds`` — Duration of the window in seconds (computed property)

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

**SunConstraint**
  Sun proximity constraint.
  
  **Constructor:**
    ``SunConstraint(min_angle=45.0)``
  
  **Attributes:**
    * ``type`` — Always "sun"
    * ``min_angle`` — Minimum angular separation from Sun in degrees (0-180)
    * ``max_angle`` — Maximum angular separation from Sun in degrees (0-180), optional

**MoonConstraint**
  Moon proximity constraint.
  
  **Constructor:**
    ``MoonConstraint(min_angle=30.0)``
  
  **Attributes:**
    * ``type`` — Always "moon"
    * ``min_angle`` — Minimum angular separation from Moon in degrees (0-180)
    * ``max_angle`` — Maximum angular separation from Moon in degrees (0-180), optional

**EarthLimbConstraint**
  Earth limb avoidance constraint.
  
  **Constructor:**
    ``EarthLimbConstraint(min_angle=10.0, include_refraction=False, horizon_dip=False)``
  
  **Attributes:**
    * ``type`` — Always "earth_limb"
    * ``min_angle`` — Minimum angular separation from Earth's limb in degrees (0-180)
    * ``max_angle`` — Maximum angular separation from Earth's limb in degrees (0-180), optional
      * ``include_refraction`` — Include atmospheric refraction correction (~0.57°) for ground observers (default: False)
      * ``horizon_dip`` — Include geometric horizon dip correction for ground observers (default: False)

**BodyConstraint**
  Solar system body proximity constraint.
  
  **Constructor:**
    ``BodyConstraint(body="Mars", min_angle=15.0)``
  
  **Attributes:**
    * ``type`` — Always "body"
    * ``body`` — Name of the solar system body (e.g., "Mars", "Jupiter")
    * ``min_angle`` — Minimum angular separation from body in degrees (0-180)
    * ``max_angle`` — Maximum angular separation from body in degrees (0-180), optional

**EclipseConstraint**
  Eclipse constraint (Earth shadow).
  
  **Constructor:**
    ``EclipseConstraint(umbra_only=True)``
  
  **Attributes:**
    * ``type`` — Always "eclipse"
    * ``umbra_only`` — If True, only umbra counts. If False, includes penumbra.

**AndConstraint**
  Logical AND combination of constraints.
  
  **Constructor:**
    ``AndConstraint(constraints=[constraint1, constraint2])``
  
  **Attributes:**
    * ``type`` — Always "and"
    * ``constraints`` — List of constraints to combine with AND

**OrConstraint**
  Logical OR combination of constraints.
  
  **Constructor:**
    ``OrConstraint(constraints=[constraint1, constraint2])``
  
  **Attributes:**
    * ``type`` — Always "or"
    * ``constraints`` — List of constraints to combine with OR

**XorConstraint**
  Logical XOR combination of constraints.
  
  Violation semantics: The XOR constraint is violated when exactly one sub-constraint is violated; it is satisfied otherwise (i.e., when either none or more than one sub-constraints are violated). This mirrors boolean XOR over "violation" states.
  
  **Constructor:**
    ``XorConstraint(constraints=[constraint1, constraint2, ...])``
  
  **Attributes:**
    * ``type`` — Always "xor"
    * ``constraints`` — List of constraints (minimum 2) evaluated with XOR violation semantics

**NotConstraint**
  Logical NOT (negation) of a constraint.
  
  **Constructor:**
    ``NotConstraint(constraint=some_constraint)``
  
  **Attributes:**
    * ``type`` — Always "not"
    * ``constraint`` — Constraint to negate

**Constraint Operators**

Constraint configurations support Python bitwise operators for convenient combination:

* ``constraint1 & constraint2`` — Logical AND (equivalent to ``AndConstraint``)
* ``constraint1 | constraint2`` — Logical OR (equivalent to ``OrConstraint``)
* ``constraint1 ^ constraint2`` — Logical XOR (equivalent to ``XorConstraint``)
* ``~constraint`` — Logical NOT (equivalent to ``NotConstraint``)

Usage examples are provided in the examples section of the docs.

Performance Notes
^^^^^^^^^^^^^^^^^

**Constraint Evaluation Optimizations**

The constraint system includes several performance optimizations for efficient evaluation:

* **Property Caching**: The ``timestamp`` and ``constraint_array`` properties on ephemeris and constraint result objects are cached for repeated access (90x+ speedup on subsequent accesses)

* **Subset Evaluation**: Use ``times`` or ``indices`` parameters in ``evaluate()`` to compute constraints for specific times only, avoiding full ephemeris evaluation

* **Single Time Checks**: For checking a single time, use ``Constraint.in_constraint()`` which is optimized for single-point evaluation

* **Optimal Usage Patterns**:

  - **FASTEST**: Evaluate once, then use ``constraint_array`` property::
  
      result = constraint.evaluate(eph, ra, dec)
      for i in range(len(result.timestamp)):
          if result.constraint_array[i]:  # ~1000x faster than alternatives
              # Target is visible at this time
              pass
  
  - **FAST**: Evaluate once, then loop over result::
  
      result = constraint.evaluate(eph, ra, dec)
      for i, time in enumerate(result.timestamp):
          if result.in_constraint(time):  # ~100x faster than evaluating each time
              # Target is visible
              pass
  
  - **SLOW (avoid)**: Calling ``in_constraint()`` in a loop::
  
      # Don't do this - evaluates ephemeris 1000s of times!
      for time in eph.timestamp:
          if constraint.in_constraint(time, eph, ra, dec):
              pass

**Timestamp Access**

All ephemeris and constraint result objects return NumPy arrays for the ``timestamp`` property, which is significantly faster than Python lists for indexing operations.

