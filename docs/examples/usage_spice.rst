Using SPICEEphemeris
====================

This example shows how to ensure a planetary SPK file is available and query
planetary positions.

.. code-block:: python

    import datetime as dt
    import numpy as np
    import rust_ephem as re

    # Ensure the planetary ephemeris is available; download if missing.
    # Uses default DE440S path and URL.
    re.ensure_planetary_ephemeris()

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create SPICE ephemeris for a specific body
    # Example: Moon (NAIF ID 301) relative to Earth center (399)
    spk_path = "path/to/your/spk/file.bsp"
    spice = re.SPICEEphemeris(
        spk_path=spk_path,
        naif_id=301,  # Moon
        begin=begin,
        end=end,
        step_size=step_size,
        center_id=399,  # Earth
        polar_motion=False
    )

    # Access pre-computed frames (PositionVelocityData objects)
    pv_gcrs = spice.gcrs_pv
    pv_itrs = spice.itrs_pv

    # Access Sun and Moon positions/velocities
    sun = spice.sun_pv
    moon = spice.moon_pv

    # Access timestamps
    times = spice.timestamp

    print("GCRS position (km):", pv_gcrs.position[0])  # First timestep
    print("Sun position norm (km):", np.linalg.norm(sun.position[0]))

    # Access astropy SkyCoord objects (requires astropy)
    gcrs_skycoord = spice.gcrs
    earth_skycoord = spice.earth
    sun_skycoord = spice.sun

SPICEEphemeris Error Handling
------------------------------
- If the SPK file is missing and ``download_if_missing=False`` was used, an
  exception is raised.
- Use ``is_planetary_ephemeris_initialized()`` to test readiness before creating
  SPICEEphemeris objects.

Additional Time System Functions
---------------------------------
.. code-block:: python

    # Check initialization status
    if re.is_planetary_ephemeris_initialized():
        print("Planetary ephemeris ready")
    
    # Initialize UT1 provider for better accuracy
    if re.init_ut1_provider():
        print("UT1 data loaded")
    
    # Initialize EOP provider for polar motion
    if re.init_eop_provider():
        print("EOP data loaded")
    
    # Get time system offsets
    when = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    tai_utc = re.get_tai_utc_offset(when)  # Leap seconds
    ut1_utc = re.get_ut1_utc_offset(when)  # UT1-UTC offset
    xp, yp = re.get_polar_motion(when)     # Polar motion in arcseconds
