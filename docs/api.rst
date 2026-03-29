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

**Ephemeris** (Abstract Base Class)
  Common interface for all ephemeris types. All concrete ephemeris classes
  (TLEEphemeris, SPICEEphemeris, GroundEphemeris, OEMEphemeris) implement this
  interface and can be used interchangeably where an ``Ephemeris`` is expected.

  Use ``isinstance(obj, Ephemeris)`` to check if an object is any ephemeris type.

  **Common Properties:**
    * ``timestamp`` — Array of UTC timestamps
    * ``gcrs_pv`` — Position/velocity in GCRS frame
    * ``itrs_pv`` — Position/velocity in ITRS frame
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``sun``, ``moon``, ``earth`` — Celestial body SkyCoord objects
    * ``sun_pv``, ``moon_pv`` — Celestial body position/velocity data
    * ``obsgeoloc``, ``obsgeovel`` — Observer location/velocity in GCRS
    * ``latitude_deg``, ``longitude_deg``, ``height_m`` — Geodetic coordinates
    * ``sun_radius_deg``, ``moon_radius_deg``, ``earth_radius_deg`` — Angular radii
    * ``sun_ra_dec_deg``, ``moon_ra_dec_deg``, ``earth_ra_dec_deg`` — RA/Dec as Nx2 arrays (cached)
    * ``sun_ra_deg``, ``sun_dec_deg``, etc. — Individual RA or Dec as 1D arrays
    * ``begin``, ``end``, ``step_size``, ``polar_motion`` — Time range properties

  **Common Methods:**
    * ``index(time)`` — Find closest timestamp index
    * ``get_body(body, spice_kernel=None, use_horizons=False)`` — Get SkyCoord for a celestial body. If ``use_horizons=True``, falls back to JPL Horizons when the body is not found in SPICE kernels.
    * ``get_body_pv(body, spice_kernel=None, use_horizons=False)`` — Get position/velocity for a celestial body. If ``use_horizons=True``, falls back to JPL Horizons when the body is not found in SPICE kernels.
    * ``moon_illumination(time_indices=None)`` — Calculate Moon illumination fraction (0.0-1.0) as seen from observer
    * ``radec_to_altaz(ra_deg, dec_deg, time_indices=None)`` — Convert RA/Dec to Alt/Az coordinates
    * ``calculate_airmass(ra_deg, dec_deg, time_indices=None)`` — Calculate astronomical airmass for target

  **Type Alias:**
    ``EphemerisType = TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris``

**TLEEphemeris**
  Propagate Two-Line Element (TLE) sets with SGP4 and convert to coordinate frames.

  **Constructor:**
    ``TLEEphemeris(tle1=None, tle2=None, begin=None, end=None, step_size=60, *, polar_motion=False, tle=None, norad_id=None, norad_name=None, spacetrack_username=None, spacetrack_password=None, epoch_tolerance_days=None, enforce_source=None)``

    **Parameters:**
      * ``tle1`` (str, optional) — First line of TLE (legacy method)
      * ``tle2`` (str, optional) — Second line of TLE (legacy method)
      * ``tle`` (str | TLERecord, optional) — Path to TLE file, URL to download TLE from, or a ``TLERecord`` object
      * ``norad_id`` (int, optional) — NORAD catalog ID to fetch TLE. If Space-Track credentials are available, Space-Track is tried first with failover to Celestrak.
      * ``norad_name`` (str, optional) — Satellite name to fetch TLE from Celestrak
      * ``begin`` (datetime) — Start time for ephemeris (required)
      * ``end`` (datetime) — End time for ephemeris (required)
      * ``step_size`` (int) — Time step in seconds (default: 60)
      * ``polar_motion`` (bool) — Apply polar motion corrections (default: False)
      * ``spacetrack_username`` (str, optional) — Space-Track.org username (or use ``SPACETRACK_USERNAME`` env var)
      * ``spacetrack_password`` (str, optional) — Space-Track.org password (or use ``SPACETRACK_PASSWORD`` env var)
      * ``epoch_tolerance_days`` (float, optional) — For Space-Track cache: how many days TLE epoch can differ from target epoch (default: 4.0 days)
      * ``enforce_source`` (str, optional) — Enforce use of specific source without failover. Must be ``"celestrak"``, ``"spacetrack"``, or ``None``

    **Notes:**
      * Must provide exactly one of: (``tle1``, ``tle2``), ``tle``, ``norad_id``, or ``norad_name``
      * ``begin`` and ``end`` parameters are required
      * File paths and URLs are cached locally for performance
      * Space-Track.org credentials can also be provided via ``.env`` file

  **Attributes (read-only):**
    * ``tle_epoch`` — TLE epoch as Python datetime (extracted from line 1)
    * ``teme_pv`` — Position/velocity in TEME frame (PositionVelocityData)
    * ``itrs_pv`` — Position/velocity in ITRS frame (PositionVelocityData)
    * ``gcrs_pv`` — Position/velocity in GCRS frame (PositionVelocityData)
    * ``sun_pv`` — Sun position/velocity in GCRS frame (PositionVelocityData)
    * ``moon_pv`` — Moon position/velocity in GCRS frame (PositionVelocityData)
    * ``timestamp`` — List of Python datetime objects
    * ``itrs`` — ITRS coordinates as astropy SkyCoord
    * ``gcrs`` — GCRS coordinates as astropy SkyCoord
    * ``latitude`` — Observatory latitude as an astropy Quantity array (degrees), one per timestamp
    * ``latitude_deg`` — Observatory latitude as NumPy array (degrees), one per timestamp
    * ``latitude_rad`` — Observatory latitude as NumPy array (radians), one per timestamp
    * ``longitude`` — Observatory longitude as an astropy Quantity array (degrees), one per timestamp
    * ``longitude_deg`` — Observatory longitude as NumPy array (degrees), one per timestamp
    * ``longitude_rad`` — Observatory longitude as NumPy array (radians), one per timestamp
    * ``height`` — Observatory height as an astropy Quantity array (meters), one per timestamp
    * ``height_m`` — Observatory height as raw NumPy array (meters), one per timestamp
    * ``height_km`` — Observatory height as raw NumPy array (kilometers), one per timestamp
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
    * ``sun_ra_dec_deg`` — Sun RA/Dec as Nx2 NumPy array (degrees), cached
    * ``sun_ra_dec_rad`` — Sun RA/Dec as Nx2 NumPy array (radians), cached
    * ``moon_ra_dec_deg`` — Moon RA/Dec as Nx2 NumPy array (degrees), cached
    * ``moon_ra_dec_rad`` — Moon RA/Dec as Nx2 NumPy array (radians), cached
    * ``earth_ra_dec_deg`` — Earth RA/Dec as Nx2 NumPy array (degrees), cached
    * ``earth_ra_dec_rad`` — Earth RA/Dec as Nx2 NumPy array (radians), cached
    * ``sun_ra_deg``, ``sun_dec_deg`` — Sun RA and Dec as separate 1D arrays (degrees)
    * ``sun_ra_rad``, ``sun_dec_rad`` — Sun RA and Dec as separate 1D arrays (radians)
    * ``moon_ra_deg``, ``moon_dec_deg`` — Moon RA and Dec as separate 1D arrays (degrees)
    * ``moon_ra_rad``, ``moon_dec_rad`` — Moon RA and Dec as separate 1D arrays (radians)
    * ``earth_ra_deg``, ``earth_dec_deg`` — Earth RA and Dec as separate 1D arrays (degrees)
    * ``earth_ra_rad``, ``earth_dec_rad`` — Earth RA and Dec as separate 1D arrays (radians)

  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime

      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

    * ``get_body_pv(body)`` — Get position/velocity of a solar system body relative to observer

      - ``body`` — Body name (e.g., "Sun", "Moon", "Mars") or NAIF ID as string (e.g., "10", "301")
      - Returns: ``PositionVelocityData`` with position/velocity in GCRS frame
      - Requires: ``ensure_planetary_ephemeris()`` called first

    * ``get_body(body)`` — Get SkyCoord for a solar system body with observer location set

      - ``body`` — Body name or NAIF ID as string
      - Returns: ``astropy.coordinates.SkyCoord`` in GCRS frame with obsgeoloc/obsgeovel set
      - Requires: ``ensure_planetary_ephemeris()`` called first

**SPICEEphemeris**
  Spacecraft ephemeris from SPICE SPK (Spacecraft and Planet Kernel) files.
  Use this for missions that provide trajectory data in SPICE format.

  **Constructor:**
    ``SPICEEphemeris(spk_path, naif_id, begin, end, step_size=60, center_id=399, *, polar_motion=False)``

    * ``spk_path`` — Path to the SPICE SPK file containing spacecraft trajectory
    * ``naif_id`` — NAIF ID of the spacecraft (typically negative, e.g., -82 for Cassini)
    * ``center_id`` — NAIF ID of the observer center (default: 399 = Earth)

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
    * ``latitude`` — Geodetic latitude as an astropy Quantity array (degrees), one per timestamp
    * ``latitude_deg`` — Geodetic latitude as NumPy array (degrees), one per timestamp
    * ``latitude_rad`` — Geodetic latitude as NumPy array (radians), one per timestamp
    * ``longitude`` — Geodetic longitude as an astropy Quantity array (degrees), one per timestamp
    * ``longitude_deg`` — Geodetic longitude as NumPy array (degrees), one per timestamp
    * ``longitude_rad`` — Geodetic longitude as NumPy array (radians), one per timestamp
    * ``height`` — Height as an astropy Quantity array (meters), one per timestamp
    * ``height_m`` — Height as raw NumPy array (meters), one per timestamp
    * ``height_km`` — Height as raw NumPy array (kilometers), one per timestamp
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
    * ``sun_ra_dec_deg`` — Sun RA/Dec as Nx2 NumPy array (degrees), cached
    * ``sun_ra_dec_rad`` — Sun RA/Dec as Nx2 NumPy array (radians), cached
    * ``moon_ra_dec_deg`` — Moon RA/Dec as Nx2 NumPy array (degrees), cached
    * ``moon_ra_dec_rad`` — Moon RA/Dec as Nx2 NumPy array (radians), cached
    * ``earth_ra_dec_deg`` — Earth RA/Dec as Nx2 NumPy array (degrees), cached
    * ``earth_ra_dec_rad`` — Earth RA/Dec as Nx2 NumPy array (radians), cached
    * ``sun_ra_deg``, ``sun_dec_deg`` — Sun RA and Dec as separate 1D arrays (degrees)
    * ``sun_ra_rad``, ``sun_dec_rad`` — Sun RA and Dec as separate 1D arrays (radians)
    * ``moon_ra_deg``, ``moon_dec_deg`` — Moon RA and Dec as separate 1D arrays (degrees)
    * ``moon_ra_rad``, ``moon_dec_rad`` — Moon RA and Dec as separate 1D arrays (radians)
    * ``earth_ra_deg``, ``earth_dec_deg`` — Earth RA and Dec as separate 1D arrays (degrees)
    * ``earth_ra_rad``, ``earth_dec_rad`` — Earth RA and Dec as separate 1D arrays (radians)

  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime

      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

    * ``get_body_pv(body)`` — Get position/velocity of a solar system body relative to observer

      - ``body`` — Body name (e.g., "Sun", "Moon", "Mars") or NAIF ID as string
      - Returns: ``PositionVelocityData`` with position/velocity in GCRS frame

    * ``get_body(body)`` — Get SkyCoord for a solar system body with observer location set

      - ``body`` — Body name or NAIF ID as string
      - Returns: ``astropy.coordinates.SkyCoord`` in GCRS frame

**GroundEphemeris**
  Ground-based observatory ephemeris for a fixed point on Earth's surface.

  **Constructor:**
    ``GroundEphemeris(latitude, longitude, height, begin, end, step_size=60, *, polar_motion=False)``

    * ``latitude`` — Geodetic latitude in degrees (-90 to 90)
    * ``longitude`` — Geodetic longitude in degrees (-180 to 180)
    * ``height`` — Altitude in meters above WGS84 ellipsoid

  **Attributes (read-only):**
    * ``latitude`` — Observatory latitude as an astropy Quantity array (degrees), one per timestamp
    * ``longitude`` — Observatory longitude as an astropy Quantity array (degrees), one per timestamp
    * ``height`` — Observatory height as an astropy Quantity array (meters), one per timestamp
    * ``height_m`` — Observatory height raw numpy array (meters), one per timestamp
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
    * ``latitude`` — Geodetic latitude as an astropy Quantity array (degrees), one per timestamp
    * ``latitude_deg`` — Geodetic latitude as NumPy array (degrees), one per timestamp
    * ``latitude_rad`` — Geodetic latitude as NumPy array (radians), one per timestamp
    * ``longitude`` — Geodetic longitude as an astropy Quantity array (degrees), one per timestamp
    * ``longitude_deg`` — Geodetic longitude as NumPy array (degrees), one per timestamp
    * ``longitude_rad`` — Geodetic longitude as NumPy array (radians), one per timestamp
    * ``height`` — Height as an astropy Quantity array (meters), one per timestamp
    * ``height_m`` — Height as raw NumPy array (meters), one per timestamp
    * ``height_km`` — Height as raw NumPy array (kilometers), one per timestamp
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
    * ``sun_ra_dec_deg`` — Sun RA/Dec as Nx2 NumPy array (degrees), cached
    * ``sun_ra_dec_rad`` — Sun RA/Dec as Nx2 NumPy array (radians), cached
    * ``moon_ra_dec_deg`` — Moon RA/Dec as Nx2 NumPy array (degrees), cached
    * ``moon_ra_dec_rad`` — Moon RA/Dec as Nx2 NumPy array (radians), cached
    * ``earth_ra_dec_deg`` — Earth RA/Dec as Nx2 NumPy array (degrees), cached
    * ``earth_ra_dec_rad`` — Earth RA/Dec as Nx2 NumPy array (radians), cached
    * ``sun_ra_deg``, ``sun_dec_deg`` — Sun RA and Dec as separate 1D arrays (degrees)
    * ``sun_ra_rad``, ``sun_dec_rad`` — Sun RA and Dec as separate 1D arrays (radians)
    * ``moon_ra_deg``, ``moon_dec_deg`` — Moon RA and Dec as separate 1D arrays (degrees)
    * ``moon_ra_rad``, ``moon_dec_rad`` — Moon RA and Dec as separate 1D arrays (radians)
    * ``earth_ra_deg``, ``earth_dec_deg`` — Earth RA and Dec as separate 1D arrays (degrees)
    * ``earth_ra_rad``, ``earth_dec_rad`` — Earth RA and Dec as separate 1D arrays (radians)

  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime

      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2024, 1, 1, 12, 0, 0))`` then ``sun_position = eph.sun_pv.position[idx]``

    * ``get_body_pv(body)`` — Get position/velocity of a solar system body relative to observer

      - ``body`` — Body name (e.g., "Sun", "Moon", "Mars") or NAIF ID as string
      - Returns: ``PositionVelocityData`` with position/velocity in GCRS frame
      - Requires: ``ensure_planetary_ephemeris()`` called first

    * ``get_body(body)`` — Get SkyCoord for a solar system body with observer location set

      - ``body`` — Body name or NAIF ID as string
      - Returns: ``astropy.coordinates.SkyCoord`` in GCRS frame

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
    * ``latitude`` — Geodetic latitude as an astropy Quantity array (degrees), one per timestamp
    * ``latitude_deg`` — Geodetic latitude as NumPy array (degrees), one per timestamp
    * ``latitude_rad`` — Geodetic latitude as NumPy array (radians), one per timestamp
    * ``longitude`` — Geodetic longitude as an astropy Quantity array (degrees), one per timestamp
    * ``longitude_deg`` — Geodetic longitude as NumPy array (degrees), one per timestamp
    * ``longitude_rad`` — Geodetic longitude as NumPy array (radians), one per timestamp
    * ``height`` — Height as an astropy Quantity array (meters), one per timestamp
    * ``height_m`` — Height as raw NumPy array (meters), one per timestamp
    * ``height_km`` — Height as raw NumPy array (kilometers), one per timestamp
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
    * ``sun_ra_dec_deg`` — Sun RA/Dec as Nx2 NumPy array (degrees), cached
    * ``sun_ra_dec_rad`` — Sun RA/Dec as Nx2 NumPy array (radians), cached
    * ``moon_ra_dec_deg`` — Moon RA/Dec as Nx2 NumPy array (degrees), cached
    * ``moon_ra_dec_rad`` — Moon RA/Dec as Nx2 NumPy array (radians), cached
    * ``earth_ra_dec_deg`` — Earth RA/Dec as Nx2 NumPy array (degrees), cached
    * ``earth_ra_dec_rad`` — Earth RA/Dec as Nx2 NumPy array (radians), cached
    * ``sun_ra_deg``, ``sun_dec_deg`` — Sun RA and Dec as separate 1D arrays (degrees)
    * ``sun_ra_rad``, ``sun_dec_rad`` — Sun RA and Dec as separate 1D arrays (radians)
    * ``moon_ra_deg``, ``moon_dec_deg`` — Moon RA and Dec as separate 1D arrays (degrees)
    * ``moon_ra_rad``, ``moon_dec_rad`` — Moon RA and Dec as separate 1D arrays (radians)
    * ``earth_ra_deg``, ``earth_dec_deg`` — Earth RA and Dec as separate 1D arrays (degrees)
    * ``earth_ra_rad``, ``earth_dec_rad`` — Earth RA and Dec as separate 1D arrays (radians)

  **Methods:**
    * ``index(time)`` — Find the index of the closest timestamp to the given datetime

      - ``time`` — Python datetime object
      - Returns: ``int`` index that can be used to access ephemeris arrays
      - Example: ``idx = eph.index(datetime(2032, 7, 1, 12, 0, 0))`` then ``position = eph.gcrs_pv.position[idx]``

    * ``get_body_pv(body)`` — Get position/velocity of a solar system body relative to observer

      - ``body`` — Body name (e.g., "Sun", "Moon", "Mars") or NAIF ID as string
      - Returns: ``PositionVelocityData`` with position/velocity in GCRS frame
      - Requires: ``ensure_planetary_ephemeris()`` called first

    * ``get_body(body)`` — Get SkyCoord for a solar system body with observer location set

      - ``body`` — Body name or NAIF ID as string
      - Returns: ``astropy.coordinates.SkyCoord`` in GCRS frame

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
    * ``Constraint.at_least(min_violated, constraints)`` — Threshold combinator (violation when at least ``min_violated`` sub-constraints are violated)
    * ``Constraint.not_(constraint)`` — Negate a constraint with logical NOT
    * ``Constraint.boresight_offset(constraint, roll_deg=0.0, roll_clockwise=False, roll_reference="north", pitch_deg=0.0, yaw_deg=0.0)`` — Wrap a constraint with fixed boresight Euler-angle offsets
      - ``roll_deg`` — Fixed instrument roll offset (degrees) relative to the spacecraft frame. Default ``0.0``. Spacecraft roll at observation time is a separate concept applied via ``target_roll`` on evaluation methods.
      - ``roll_reference`` — Roll-zero reference axis. Default is ``"north"`` (celestial-north-projected +Z zero-roll). Use ``"sun"`` for Sun-projected +Z zero-roll when needed.
    * ``Constraint.from_json(json_str)`` — Create constraint from JSON configuration

  **Methods:**
    * ``evaluate(ephemeris, target_ra, target_dec, times=None, indices=None)`` — Evaluate constraint against ephemeris data

      - ``ephemeris`` — TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris object
      - ``target_ra`` — Target right ascension in degrees (ICRS/J2000)
      - ``target_dec`` — Target declination in degrees (ICRS/J2000)
      - ``times`` — Optional: specific datetime(s) to evaluate (must exist in ephemeris)
      - ``indices`` — Optional: specific time index/indices to evaluate
      - Returns: ``ConstraintResult`` object

    * ``in_constraint_batch(ephemeris, target_ras, target_decs, times=None, indices=None)`` — **[Recommended]** Vectorized batch evaluation for multiple targets

      - ``ephemeris`` — TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris object
      - ``target_ras`` — List/array of target right ascensions in degrees (ICRS/J2000)
      - ``target_decs`` — List/array of target declinations in degrees (ICRS/J2000)
      - ``times`` — Optional: specific datetime(s) to evaluate (must exist in ephemeris)
      - ``indices`` — Optional: specific time index/indices to evaluate
      - Returns: 2D NumPy boolean array of shape (n_targets, n_times) where True indicates constraint violation
      - **Performance**: 3-50x faster than calling ``evaluate()`` in a loop
      - **Optimized**: Uses vectorized operations for batch RA/Dec conversion and constraint evaluation
      - **All constraint types supported**: Sun/Moon proximity, Earth limb, Eclipse, Body proximity, boresight offsets, and logical combinators (AND, OR, XOR, AT_LEAST, NOT)

    * ``in_constraint(time, ephemeris, target_ra, target_dec)`` — Check if target is in-constraint at a single time

      - ``time`` — Python datetime object (must exist in ephemeris timestamps)
      - Returns: ``bool`` (True if constraint is violated / target is blocked, False if satisfied)

    * ``instantaneous_field_of_regard(ephemeris, time=None, index=None, n_points=DEFAULT_N_POINTS, n_roll_samples=DEFAULT_N_ROLL_SAMPLES, target_roll=None)`` — Compute instantaneous visible sky solid angle. When ``target_roll`` is not specified, sweeps ``n_roll_samples`` spacecraft roll angles for boresight-offset constraints with non-zero pitch/yaw, giving the total accessible sky over all roll states.

      - ``ephemeris`` — TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris object
      - ``time`` — Optional datetime to evaluate (must exist in ephemeris)
      - ``index`` — Optional ephemeris index to evaluate
      - ``n_points`` — Number of sky samples (Fibonacci sphere integration, default :data:`DEFAULT_N_POINTS`)
      - ``target_roll`` — Spacecraft roll angle (degrees) to evaluate at. When ``None`` (default), sweeps all roll angles for boresight-offset FoR.
      - ``n_roll_samples`` — Spacecraft roll angles to sweep when ``target_roll`` is ``None`` and pitch/yaw offsets are present (default :data:`DEFAULT_N_ROLL_SAMPLES` = 72, i.e. 5° resolution). Ignored otherwise.
      - Returns: ``float`` steradians in ``[0, 4π]``
      - Requirement: exactly one of ``time`` or ``index`` must be provided
      - Semantics: constraints are violated when ``True``; this method integrates visible sky where constraint is ``False``

    * ``to_json()`` — Get constraint configuration as JSON string
    * ``to_dict()`` — Get constraint configuration as Python dictionary

**ConstraintResult**
  Result of constraint evaluation containing violation information.

  **Attributes (read-only):**
    * ``violations`` — List of ``ConstraintViolation`` objects
    * ``all_satisfied`` — Boolean indicating if constraint was satisfied for entire time range
    * ``constraint_name`` — String name/description of the constraint
    * ``timestamp`` — NumPy array of Python datetime objects (optimized with caching)
    * ``constraint_array`` — NumPy boolean array where True means constraint violated / target blocked (optimized with caching)
    * ``visibility`` — List of ``VisibilityWindow`` objects for contiguous satisfied periods

  **Methods:**
    * ``total_violation_duration()`` — Get total duration of violations in seconds
    * ``in_constraint(time)`` — Check if constraint is violated at a given time

      - ``time`` — Python datetime object (must exist in result timestamps)
      - Returns: ``bool`` (True if violated / target blocked, False if satisfied)

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

**TLE Fetching**

* ``fetch_tle(*, tle=None, norad_id=None, norad_name=None, epoch=None, spacetrack_username=None, spacetrack_password=None, epoch_tolerance_days=None, enforce_source=None)`` — Fetch a TLE from various sources.

  This function provides a unified interface for retrieving TLE data from local files,
  URLs, Celestrak, or Space-Track.org. Returns a ``TLERecord`` object containing the
  TLE data and metadata.

  **Parameters:**
    * ``tle`` (str, optional) — Path to TLE file or URL to download TLE from
    * ``norad_id`` (int, optional) — NORAD catalog ID to fetch TLE. If Space-Track credentials are available, Space-Track is tried first with failover to Celestrak.
    * ``norad_name`` (str, optional) — Satellite name to fetch TLE from Celestrak
    * ``epoch`` (datetime, optional) — Target epoch for Space-Track queries. If not specified, current time is used. Space-Track will fetch the TLE with epoch closest to this time.
    * ``spacetrack_username`` (str, optional) — Space-Track.org username (or use ``SPACETRACK_USERNAME`` env var)
    * ``spacetrack_password`` (str, optional) — Space-Track.org password (or use ``SPACETRACK_PASSWORD`` env var)
    * ``epoch_tolerance_days`` (float, optional) — For Space-Track cache: how many days TLE epoch can differ from target epoch (default: 4.0 days)
    * ``enforce_source`` (str, optional) — Enforce use of specific source without failover. Must be ``"celestrak"``, ``"spacetrack"``, or ``None`` (default behavior with failover)

  **Returns:**
    ``TLERecord`` — A Pydantic model containing the TLE data and metadata

  **Raises:**
    ``ValueError`` — If no valid TLE source is specified or fetching fails

  **Examples:**

  .. code-block:: python

      import rust_ephem

      # Fetch from Celestrak by NORAD ID
      tle = rust_ephem.fetch_tle(norad_id=25544)  # ISS
      print(tle.name)
      print(tle.line1)
      print(tle.line2)

      # Fetch from file
      tle = rust_ephem.fetch_tle(tle="path/to/satellite.tle")

      # Fetch from URL
      tle = rust_ephem.fetch_tle(tle="https://celestrak.org/NORAD/elements/gp.php?CATNR=25544")

      # Fetch from Space-Track with explicit credentials
      from datetime import datetime, timezone
      tle = rust_ephem.fetch_tle(
          norad_id=25544,
          spacetrack_username="your_username",
          spacetrack_password="your_password",
          epoch=datetime(2020, 1, 1, tzinfo=timezone.utc)
      )

      # Use TLERecord with TLEEphemeris
      ephem = rust_ephem.TLEEphemeris(
          tle=tle,  # Pass TLERecord directly
          begin=datetime(2024, 1, 1, tzinfo=timezone.utc),
          end=datetime(2024, 1, 2, tzinfo=timezone.utc),
          step_size=60
      )

  **Notes:**
    * Must provide exactly one of: ``tle``, ``norad_id``, or ``norad_name``
    * File paths and URLs are cached locally for improved performance
    * Space-Track.org requires free account registration at https://www.space-track.org
    * Credentials can be provided via:

      1. Explicit parameters: ``spacetrack_username`` and ``spacetrack_password``
      2. Environment variables: ``SPACETRACK_USERNAME`` and ``SPACETRACK_PASSWORD``
      3. ``.env`` file in the current directory or home directory (``~/.env``)
         containing the same environment variables

Data Models
^^^^^^^^^^^

**TLERecord**
  A Pydantic model representing a Two-Line Element (TLE) record with metadata.
  Can be passed directly to ``TLEEphemeris`` via the ``tle`` parameter.
  Supports JSON serialization for storage and transmission.

  **Attributes:**
    * ``line1`` (str) — First line of the TLE (starts with '1')
    * ``line2`` (str) — Second line of the TLE (starts with '2')
    * ``name`` (str | None) — Optional satellite name (from 3-line TLE format)
    * ``epoch`` (datetime) — TLE epoch timestamp (extracted from line1)
    * ``source`` (str | None) — Source of the TLE data (e.g., 'celestrak', 'spacetrack', 'file', 'url')

  **Computed Properties:**
    * ``norad_id`` (int) — NORAD catalog ID extracted from line1
    * ``classification`` (str) — Classification from line1 (U=unclassified, C=classified, S=secret)
    * ``international_designator`` (str) — International designator extracted from line1

  **Methods:**
    * ``to_tle_string()`` — Convert to a 2-line or 3-line TLE string format
    * ``model_dump()`` — Convert to dictionary (Pydantic)
    * ``model_dump_json()`` — Convert to JSON string (Pydantic)

  **Example:**

  .. code-block:: python

      import rust_ephem

      tle = rust_ephem.fetch_tle(norad_id=25544)

      # Access TLE data
      print(f"Satellite: {tle.name}")
      print(f"NORAD ID: {tle.norad_id}")
      print(f"Epoch: {tle.epoch}")
      print(f"Source: {tle.source}")

      # Get TLE as string
      print(tle.to_tle_string())

      # Serialize to JSON
      json_str = tle.model_dump_json()

      # Pass directly to TLEEphemeris
      ephem = rust_ephem.TLEEphemeris(
          tle=tle,
          begin=datetime(2024, 1, 1, tzinfo=timezone.utc),
          end=datetime(2024, 1, 2, tzinfo=timezone.utc)
      )

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
  Eclipse constraint (Earth shadow). Assumes an Earth-centered ephemeris (Earth at origin).

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

**AtLeastConstraint**
  Threshold (k-of-n) combination of constraints.

  Violation semantics: The AT_LEAST constraint is violated when the number of
  violated sub-constraints is greater than or equal to ``min_violated``.

  **Constructor:**
    ``AtLeastConstraint(min_violated=2, constraints=[constraint1, constraint2, ...])``

  **Attributes:**
    * ``type`` — Always "at_least"
    * ``min_violated`` — Minimum number of violated sub-constraints required (>=1)
    * ``constraints`` — List of constraints (minimum 1)

**NotConstraint**
  Logical NOT (negation) of a constraint.

  **Constructor:**
    ``NotConstraint(constraint=some_constraint)``

  **Attributes:**
    * ``type`` — Always "not"
    * ``constraint`` — Constraint to negate

**Common Constraint Methods**

All constraint configuration classes (SunConstraint, MoonConstraint, etc.) inherit these methods from ``RustConstraintMixin``:

* ``evaluate(ephemeris, target_ra, target_dec, times=None, indices=None)`` — Evaluate constraint for a single target

  - Returns: ``ConstraintResult`` object
  - See ``Constraint.evaluate()`` above for parameter details

* ``in_constraint(time, ephemeris, target_ra, target_dec)`` — Check if target satisfies constraint at a single time

  - Returns: ``bool``

* ``at_least(min_violated, *others)`` — Build an ``AtLeastConstraint`` from this constraint plus additional constraints

  - Returns: ``AtLeastConstraint``

**Constraint Operators**

Constraint configurations support Python bitwise operators for convenient combination:

* ``constraint1 & constraint2`` — Logical AND (equivalent to ``AndConstraint``)
* ``constraint1 | constraint2`` — Logical OR (equivalent to ``OrConstraint``)
* ``constraint1 ^ constraint2`` — Logical XOR (equivalent to ``XorConstraint``)
* ``~constraint`` — Logical NOT (equivalent to ``NotConstraint``)

Usage examples are provided in the examples section of the docs.

**Type Aliases**

``ConstraintConfig``
  Union type for all constraint configuration classes::

    ConstraintConfig = (
        SunConstraint | MoonConstraint | EclipseConstraint |
        EarthLimbConstraint | BodyConstraint | AndConstraint |
      OrConstraint | XorConstraint | AtLeastConstraint | NotConstraint
    )

``CombinedConstraintConfig``
  Pydantic TypeAdapter for parsing constraint configurations from JSON.

Performance Notes
^^^^^^^^^^^^^^^^^

**Constraint Evaluation Optimizations**

The constraint system includes several performance optimizations for efficient evaluation:

* **Property Caching**: The ``timestamp`` and ``constraint_array`` properties on ephemeris and constraint result objects are cached for repeated access (90x+ speedup on subsequent accesses)

* **Subset Evaluation**: Use ``times`` or ``indices`` parameters in ``evaluate()`` to compute constraints for specific times only, avoiding full ephemeris evaluation

* **Single Time Checks**: For checking a single time, use ``Constraint.in_constraint()`` which is optimized for single-point evaluation

* **Vectorized Batch Evaluation**: For checking multiple targets, use ``in_constraint_batch()`` which evaluates all targets in a single call, eliminating Python call overhead

* **Optimal Usage Patterns**:

  - **FASTEST (Multiple Targets)**: Use batch evaluation for multiple targets::

      target_ras = [0.0, 90.0, 180.0, 270.0]
      target_decs = [0.0, 30.0, -30.0, 60.0]
      violations = constraint.in_constraint_batch(eph, target_ras, target_decs)
      # violations is (n_targets, n_times) array
      # violations[i, j] = True if target i violates constraint at time j

  - **FAST (Single Target)**: Evaluate once, then use ``constraint_array`` property::

      result = constraint.evaluate(eph, ra, dec)
      for i in range(len(result.timestamp)):
          if result.constraint_array[i]:  # ~1000x faster than alternatives
              # Target is visible at this time
              pass

  - **OK**: Evaluate once, then loop over result::

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
