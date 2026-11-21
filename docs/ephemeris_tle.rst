Using TLEEphemeris
==================

This example shows how to propagate a satellite from a Two-Line Element (TLE)
set and obtain positions in different frames.

TLE propagation uses the WGS-72 geopotential constants and improved SGP4
operation mode, matching the defaults used by Python SGP4 and Skyfield. The
AFSPC TLE epoch and sidereal-time expressions are used when initializing the
propagator. These choices are independent of the WGS-84 ellipsoid used for
geodetic latitude, longitude, and height.

``TLEEphemeris`` exposes these choices through the read-only ``propagator``,
``gravity_model``, and ``sgp4_operation_mode`` properties so serialized
products can record the actual implementation rather than duplicating labels.

.. code-block:: python

    import datetime as dt
    import numpy as np
    import rust_ephem as re

    # Example TLE (ISS, may be outdated)
    tle1 = "1 25544U 98067A   20344.91777778  .00002182  00000-0  46906-4 0  9991"
    tle2 = "2 25544  51.6460  44.6055 0002398  79.4451  23.5248 15.49364984256518"

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create ephemeris from TLE - several methods available:

    # Method 1: Direct TLE strings (legacy)
    sat = re.TLEEphemeris(tle1, tle2, begin, end, step_size, polar_motion=False)

    # Method 2: From file path
    # sat = re.TLEEphemeris(tle="path/to/tle_file.txt", begin=begin, end=end, step_size=step_size)

    # Method 3: From URL (with caching)
    # sat = re.TLEEphemeris(tle="https://celestrak.org/NORAD/elements/gp.php?CATNR=25544", begin=begin, end=end, step_size=step_size)

    # Method 4: From NORAD ID (Celestrak, or Space-Track.org if credentials are set)
    # If SPACETRACK_USERNAME and SPACETRACK_PASSWORD are set, Space-Track.org is
    # tried first with automatic failover to Celestrak on failure.
    # sat = re.TLEEphemeris(norad_id=25544, begin=begin, end=end, step_size=step_size)

    # Method 5: From satellite name (fetches from Celestrak)
    # sat = re.TLEEphemeris(norad_name="ISS (ZARYA)", begin=begin, end=end, step_size=step_size)

    # Method 6: Explicit Space-Track.org credentials with norad_id
    # sat = re.TLEEphemeris(
    #     norad_id=25544,
    #     spacetrack_username="your_username",
    #     spacetrack_password="your_password",
    #     begin=begin, end=end, step_size=step_size,
    #     epoch_tolerance_days=4.0  # Optional: cache tolerance in days
    # )

    # All frames are pre-computed during initialization
    # Access pre-computed frames (PositionVelocityData objects)
    pv_teme = sat.teme_pv
    pv_itrs = sat.itrs_pv
    pv_gcrs = sat.gcrs_pv

    # Access Sun and Moon positions/velocities
    sun = sat.sun_pv
    moon = sat.moon_pv

    # Access timestamps
    times = sat.timestamp

    # Position (km) and velocity (km/s)
    print("TEME position (km):", pv_teme.position[0])  # First timestep
    print("TEME velocity (km/s):", pv_teme.velocity[0])
    print("GCRS position norm (km):", np.linalg.norm(pv_gcrs.position[0]))

    # Access astropy SkyCoord objects (requires astropy)
    gcrs_skycoord = sat.gcrs
    itrs_skycoord = sat.itrs

    # Access obsgeoloc/obsgeovel for astropy GCRS frames
    obsgeoloc = sat.obsgeoloc
    obsgeovel = sat.obsgeovel

    # Geodetic coordinates for the observer (derived from positions)
    # These are Quantity arrays — index [0] gives the scalar at first timestep
    print("Latitude (deg):", sat.latitude_deg[0])
    print("Longitude (deg):", sat.longitude_deg[0])
    print("Height (m):", sat.height_m[0])

States at arbitrary times
-------------------------

``gcrs_pv``, ``sun_pv`` and ``moon_pv`` hold states on the ephemeris grid. To get
states between grid points, every ephemeris type provides ``gcrs_pv_at``,
``itrs_pv_at``, ``sun_pv_at`` and ``moon_pv_at``. Grid times return the stored
rows unchanged; other times use the same cubic Hermite interpolation (position
and velocity over the bracketing pair) that constraint evaluation uses for
``times=``, so geometry computed from these states matches what constraints see.
No extrapolation is done: times outside the span raise ``ValueError``.
Times may be ``datetime`` objects or a NumPy ``datetime64`` array (read as
UTC), as for the ``times=`` argument of the constraint methods.

.. code-block:: python

    t = [begin + dt.timedelta(seconds=90), begin]   # any order, duplicates allowed
    sc = sat.gcrs_pv_at(t)       # PositionVelocityData, rows follow ``t``
    sun = sat.sun_pv_at(t)
    moon = sat.moon_pv_at(t)
    sun_from_sc = sun.position - sc.position   # km, GCRS

``get_body_pv`` returns the state of a body relative to the observer directly,
and takes the same ``times``. With ``apparent=True`` the position points where
the body is seen: the light left the body one light time earlier, is bent by
the Sun, and is aberrated by the observer's barycentric velocity (about 20
arcseconds for the Sun). ``apparent_target_at`` applies the matching correction
to catalogue targets. Together they give the angles that constraints report
with ``apparent=True``:

.. code-block:: python

    import numpy as np

    sun = sat.get_body_pv("sun", times=t, apparent=True)     # km and km/s, GCRS axes
    target = sat.apparent_target_at(83.63, 22.01, t)         # (N, 3) unit vectors
    cosine = np.sum(sun.position * target, axis=1) / np.linalg.norm(sun.position, axis=1)
    sun_angle_deg = np.degrees(np.arccos(cosine))
    # equals SunConstraint(...).evaluate_elementwise(..., apparent=True)
    #        .constraint_values["sun_angle_deg"]

The length of an apparent position is the light-time distance: from the
observer at the requested time to the body where it was when the light left it.
``body`` may also be ``"moon"``, ``"earth"``, or any body the loaded kernels
cover (``"jupiter barycenter"``, ``"499"``). ``apparent=True`` needs a planetary
kernel (``ensure_planetary_ephemeris()``).

.. note::

   With ``times=`` or ``apparent=True``, and no ``spice_kernel``, the Sun and
   Moon are the states the ephemeris computed when it was built: the ones
   ``sun_pv_at``, ``moon_pv_at`` and constraints use. If no planetary kernel
   was loaded or cached at that moment they are analytic (the Moon is off by
   up to about 200 km, a few arcseconds), and they stay analytic after a
   kernel is loaded. The plain ``get_body_pv(body)`` and ``get_body(body)``
   read the kernel when they are called, so the two then disagree. A
   ``UserWarning`` reports it. Load the kernel before building the ephemeris,
   or pass ``spice_kernel=`` to read it for one call.

``ras``, ``decs`` and ``times`` must be single values or one-dimensional;
declinations must lie between -90 and 90 degrees and every coordinate must be
finite.

The apparent velocity is the rate of change of the apparent position. It is
not the geometric velocity: from low Earth orbit the Sun's aberration shift is
a few thousand km at 1 au and swings round with the orbital velocity, which
adds several km/s. It is found by differencing the apparent position over half
a second, so it inherits the observer's acceleration from the interpolated
ephemeris, and the aberration multiplies an error in that acceleration by the
light time to the body: about 500 s for the Sun, 1.3 s for the Moon, and 2,000
to 3,000 s for Jupiter.

How good the acceleration is depends on the ephemeris:

- Where the stored velocities are the time derivative of the stored positions
  (OEM, SPICE and file ephemerides from a numerical propagator, and ground
  sites), a one-minute step in low orbit gives about 2 m/s for the Sun and
  2 cm/s for the Moon and Earth. A finer step is better, and a more distant
  body is worse in proportion to its light time.
- For a ``TLEEphemeris`` it is worse, and a finer step does not help. SGP4's
  velocity differs from the derivative of its position by about 0.02 m/s near
  the TLE's epoch and by 0.3 to 0.5 m/s for an element set years old. The
  interpolation turns a difference :math:`\delta v` into an acceleration error
  of about :math:`6\,\delta v / \mathrm{step}`, so the velocity error is about
  :math:`(D/c)\,6\,\delta v / \mathrm{step}` for a body at distance :math:`D`,
  and grows as the step shrinks. Measured over two hours of a low orbit,
  worst case:

  .. list-table::
     :header-rows: 1

     * - Body, TLE age
       - 10 s step
       - 60 s step
       - 300 s step
     * - Sun, at epoch
       - 5 m/s
       - 2 m/s
       - 42 m/s
     * - Sun, 15 years old
       - 77 m/s
       - 16 m/s
       - 41 m/s
     * - Moon, at epoch
       - 2 cm/s
       - 2 cm/s
       - 2.5 m/s
     * - Moon, 15 years old
       - 0.4 m/s
       - 0.3 m/s
       - 2.5 m/s
     * - Jupiter, at epoch
       - 20 m/s
       - 9 m/s
       - 200 m/s

  Around one minute is the best step for a TLE observer; at 300 s the
  interpolation of the orbit itself dominates.

Apparent positions and directions do not depend on the observer's
acceleration, so none of this affects them.

TLEEphemeris Notes
------------------
- Position magnitudes should be in LEO range (6500–8000 km); velocity around
  7–8 km/s.
- All coordinate frames are pre-computed during initialization for efficiency.
- The ``polar_motion`` parameter enables polar motion corrections (requires EOP data).
- TLE data can be provided in multiple ways: direct strings, file paths, URLs, NORAD IDs, satellite names, or Space-Track.org.
- File and URL TLE sources are cached locally for improved performance on subsequent uses.
- Space-Track.org fetches use epoch-aware caching: cached TLEs are reused if their epoch is within the configured tolerance (default: ±4 days) of the requested begin time.
- See tests under ``tests/`` for more examples and validation.

Space-Track.org Integration
---------------------------

When Space-Track.org credentials are available, the ``norad_id`` parameter will:

1. Try fetching from Space-Track.org first (with epoch-based queries)
2. Fall back to Celestrak automatically if Space-Track.org fails

Credentials can be provided via:

1. Explicit parameters: ``spacetrack_username`` and ``spacetrack_password``
2. Environment variables: ``SPACETRACK_USERNAME`` and ``SPACETRACK_PASSWORD``
3. ``.env`` file in the current directory or home directory

If no credentials are found, ``norad_id`` uses Celestrak directly.

Space-Track.org provides historical TLE data with epoch-based queries. When you specify
a ``begin`` time, the library fetches the TLE with an epoch closest to that time.
This ensures the most accurate propagation for your time range.

The ``epoch_tolerance_days`` parameter controls caching behavior. If a cached TLE
exists with an epoch within ±N days of the requested begin time, it will be used
instead of making a new API request. Default tolerance is 4 days.

**Note:** Please follow Space-Track.org's `usage guidelines <https://www.space-track.org/documentation#api>`_
(max 1 TLE query per hour for automated scripts). TLEs are cached in ``~/.cache/rust_ephem/spacetrack_cache/``.

Using fetch_tle for TLE Management
----------------------------------

For more control over TLE fetching and inspection, use the ``fetch_tle()`` function
which returns a ``TLERecord`` object. This is useful when you need to:

- Inspect TLE metadata before creating an ephemeris
- Cache or store TLEs for later use
- Access TLE fields like NORAD ID, epoch, or classification

.. code-block:: python

    import rust_ephem
    from datetime import datetime, timezone

    # Fetch TLE and inspect metadata
    tle = rust_ephem.fetch_tle(norad_id=25544)

    print(f"Satellite: {tle.name}")
    print(f"NORAD ID: {tle.norad_id}")
    print(f"TLE Epoch: {tle.epoch}")
    print(f"Source: {tle.source}")
    print(f"Classification: {tle.classification}")

    # View the raw TLE lines
    print(tle.to_tle_string())

    # Pass TLERecord directly to TLEEphemeris
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    sat = rust_ephem.TLEEphemeris(
        tle=tle,  # Pass TLERecord directly
        begin=begin,
        end=end,
        step_size=60
    )

    # Access the exact TLERecord used for propagation
    used_tle = sat.tle_record
    print(used_tle.source, used_tle.epoch)

**Benefits of using fetch_tle:**

- **Metadata access**: Inspect TLE epoch, source, and satellite name before propagation
- **Serialization**: ``TLERecord`` is a Pydantic model supporting JSON serialization via ``model_dump_json()``
- **Validation**: Verify the TLE was fetched from the expected source
- **Reuse**: Fetch once, create multiple ephemeris objects with different time ranges

TLERecord Derived Fields
------------------------

``TLERecord`` exposes parsed and derived element fields populated when the
model validates ``line1`` and ``line2``. These provide direct access to
commonly used TLE values without manual column slicing.

Available helpers include:

- ``mean_motion_dot_rev_per_day2``
- ``mean_motion_ddot_rev_per_day3``
- ``bstar_drag``
- ``ephemeris_type``
- ``element_set_number``
- ``revolution_number_at_epoch``
- ``inclination_deg``
- ``right_ascension_deg``
- ``eccentricity``
- ``arg_periapsis_deg``
- ``mean_anomaly_deg``
- ``mean_motion_rev_per_day``
- ``mean_motion_rad_s``
- ``true_anomaly_deg``
- ``semimajor_axis_m`` (derived using WGS72 Earth gravitational parameter)

.. code-block:: python

    tle = rust_ephem.fetch_tle(norad_id=25544)

    print(tle.inclination_deg)
    print(tle.mean_motion_rev_per_day)
    print(tle.mean_motion_rad_s)
    print(tle.semimajor_axis_m)

For convenience, ``classical_elements(mu_m3_s2=...)`` returns the same element
set as a dictionary. When ``mu_m3_s2`` is overridden, ``SemimajorAxis_m`` is
recomputed using the supplied gravitational parameter.

.. code-block:: python

    # Serialize TLE for storage
    json_str = tle.model_dump_json()

    # Fetch from Space-Track with specific epoch
    historical_tle = rust_ephem.fetch_tle(
        norad_id=25544,
        epoch=datetime(2020, 6, 15, tzinfo=timezone.utc),
        enforce_source="spacetrack"  # Don't fall back to Celestrak
    )

See :doc:`api` for complete ``fetch_tle()`` and ``TLERecord`` documentation.
