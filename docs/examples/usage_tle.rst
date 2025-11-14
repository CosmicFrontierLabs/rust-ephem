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

    # Create ephemeris from TLE
    # All frames are pre-computed during initialization
    sat = re.TLEEphemeris(tle1, tle2, begin, end, step_size, polar_motion=False)

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

TLEEphemeris Notes
------------------
- Position magnitudes should be in LEO range (6500–8000 km); velocity around
  7–8 km/s.
- All coordinate frames are pre-computed during initialization for efficiency.
- The ``polar_motion`` parameter enables polar motion corrections (requires EOP data).
- See tests under ``tests/`` for more examples and validation.
