Using TLEEphemeris
==================

This example shows how to propagate a satellite from a Two-Line Element (TLE)
set and obtain positions in different frames.

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

**Benefits of using fetch_tle:**

- **Metadata access**: Inspect TLE epoch, source, and satellite name before propagation
- **Serialization**: ``TLERecord`` is a Pydantic model supporting JSON serialization via ``model_dump_json()``
- **Validation**: Verify the TLE was fetched from the expected source
- **Reuse**: Fetch once, create multiple ephemeris objects with different time ranges

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
