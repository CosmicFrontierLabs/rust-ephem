Using GroundEphemeris
=====================

This example shows how to compute ephemeris for a ground-based observatory at a
fixed location on Earth's surface.

.. code-block:: python

    import datetime as dt
    import numpy as np
    import rust_ephem as re

    # Define observatory location (geodetic coordinates)
    # Example: Apache Point Observatory, New Mexico
    latitude = 32.7797   # degrees
    longitude = -105.8204  # degrees
    height = 2788.0      # meters above WGS84 ellipsoid

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create ground ephemeris
    # All frames are pre-computed during initialization
    observatory = re.GroundEphemeris(
        latitude=latitude,
        longitude=longitude,
        height=height,
        begin=begin,
        end=end,
        step_size=step_size,
        polar_motion=True  # Enable polar motion corrections
    )

    # Access observatory location properties
    print("Latitude:", observatory.latitude, "degrees")
    print("Longitude:", observatory.longitude, "degrees")
    print("Height:", observatory.height, "meters")

    # Access pre-computed frames (PositionVelocityData objects)
    pv_itrs = observatory.itrs_pv  # Earth-fixed frame
    pv_gcrs = observatory.gcrs_pv       # Celestial frame

    # Access Sun and Moon positions/velocities in GCRS PositionVelocityData
    sun = observatory.sun_pv
    moon = observatory.moon_pv
    # Access timestamps
    times = observatory.timestamp

    # Observatory position and velocity
    print("ITRS position (km):", pv_itrs.position[0])  # First timestep
    print("ITRS velocity (km/s):", pv_itrs.velocity[0])  # Due to Earth rotation
    print("GCRS position (km):", pv_gcrs.position[0])

    # Sun and Moon positions
    print("Sun position (km):", sun.position[0])
    print("Moon position (km):", moon.position[0])

    # Access astropy SkyCoord objects (requires astropy)
    gcrs_skycoord = observatory.gcrs
    itrs_skycoord = observatory.itrs
    sun_skycoord = observatory.sun
    moon_skycoord = observatory.moon

    # Access obsgeoloc/obsgeovel for astropy GCRS frames
    # These are compatible with astropy.coordinates.GCRS frame parameters
    obsgeoloc = observatory.obsgeoloc  # Observatory location in GCRS
    obsgeovel = observatory.obsgeovel  # Observatory velocity in GCRS

GroundEphemeris Use Cases
--------------------------
- Computing visibility windows from ground stations
- Calculating topocentric coordinates for satellite tracking
- Generating observatory-centric Sun/Moon ephemerides
- Providing location data for astropy coordinate transformations

GroundEphemeris Notes
---------------------
- The observatory position is fixed in ITRS (Earth-fixed frame)
- Velocity in ITRS is due to Earth's rotation
- Position in GCRS changes over time due to Earth's rotation and precession
- The ``polar_motion`` parameter enables polar motion corrections (requires EOP data)
- Use ``re.init_eop_provider()`` to load EOP data for polar motion corrections
- See tests under ``tests/test_ground_ephemeris.py`` for more examples
