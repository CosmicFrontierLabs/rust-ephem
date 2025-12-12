Using SPICEEphemeris
====================

``SPICEEphemeris`` loads spacecraft trajectory data from SPICE SPK files. This is
the standard format for NASA and ESA mission ephemeris data.

.. note::
   ``SPICEEphemeris`` is for **spacecraft** trajectories stored in SPK files.
   To query planetary body positions (Sun, Moon, planets), use ``get_body()``
   or ``get_body_pv()`` on any ephemeris type.

Basic Usage
-----------

.. code-block:: python

    import datetime as dt
    import numpy as np
    import rust_ephem as re

    # Initialize planetary ephemeris for Sun/Moon positions
    re.ensure_planetary_ephemeris()

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create SPICE ephemeris for your spacecraft
    # You need an SPK file containing your spacecraft's trajectory
    spk_path = "path/to/your/spacecraft.bsp"
    spice = re.SPICEEphemeris(
        spk_path=spk_path,
        naif_id=-12345,   # Your spacecraft's NAIF ID (typically negative)
        begin=begin,
        end=end,
        step_size=step_size,
        center_id=399,    # Observer center: Earth (default)
        polar_motion=False
    )

    # Access spacecraft positions (PositionVelocityData objects)
    pv_gcrs = spice.gcrs_pv
    pv_itrs = spice.itrs_pv

    # Access Sun and Moon positions relative to spacecraft
    sun = spice.sun_pv
    moon = spice.moon_pv

    # Access timestamps
    times = spice.timestamp

    print("Spacecraft GCRS position (km):", pv_gcrs.position[0])
    print("Distance from Earth (km):", np.linalg.norm(pv_gcrs.position[0]))

    # Access astropy SkyCoord objects
    gcrs_skycoord = spice.gcrs
    earth_skycoord = spice.earth
    sun_skycoord = spice.sun

    # Geodetic coordinates (lat/lon/height) of spacecraft sub-point
    print("Sub-satellite latitude (deg):", spice.latitude_deg[0])
    print("Sub-satellite longitude (deg):", spice.longitude_deg[0])
    print("Altitude (m):", spice.height_m[0])

SPICEEphemeris Error Handling
-----------------------------

- If the SPK file doesn't exist or doesn't contain data for the requested NAIF ID
  and time range, an exception is raised.
- Use ``is_planetary_ephemeris_initialized()`` to check if planetary ephemeris
  (for Sun/Moon) is ready before accessing ``.sun`` or ``.moon`` properties.

Where to Get SPK Files
----------------------

Spacecraft SPK files are available from:

- **NASA NAIF**: https://naif.jpl.nasa.gov/naif/data.html
- **ESA SPICE Service**: https://www.cosmos.esa.int/web/spice
- **Mission-specific archives** (PDS, mission websites)

Common spacecraft NAIF IDs:

- -82: Cassini
- -98: New Horizons
- -143: Mars Reconnaissance Orbiter

Check your SPK file's documentation for the correct NAIF ID.

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
