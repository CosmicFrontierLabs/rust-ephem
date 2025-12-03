
Astropy SkyCoord Integration
-----------------------------

The library provides direct access to astropy SkyCoord objects for all coordinate
frames and celestial bodies. This is **84x faster** than manual Python loops for
coordinate conversion.

**Satellite Position Frames**

.. code-block:: python

    import rust_ephem
    from datetime import datetime, timezone

    # Create ephemeris
    tle1 = "1 25544U 98067A   20344.91777778  .00002182  00000-0  46906-4 0  9991"
    tle2 = "2 25544  51.6460  44.6055 0002398  79.4451  23.5248 15.49364984256518"
    
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    
    ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=60)
    
    # Access satellite positions as SkyCoord objects
    gcrs_skycoord = ephem.gcrs  # GCRS frame (celestial)
    itrs_skycoord = ephem.itrs  # ITRS frame (Earth-fixed)
    teme_skycoord = ephem.teme     # TEME frame (SGP4 output)
    
    # Each SkyCoord includes position, velocity, and proper frame
    print(f"GCRS position: {gcrs_skycoord[0].cartesian.xyz}")
    print(f"GCRS velocity: {gcrs_skycoord[0].velocity.d_xyz}")
    print(f"ITRS position: {itrs_skycoord[0].cartesian.xyz}")

**Sun, Moon, and Earth Positions**

All celestial body positions include the spacecraft/observatory location as
``obsgeoloc`` and ``obsgeovel`` in their GCRS frame:

.. code-block:: python

    # Get Sun position as seen from spacecraft
    sun_skycoord = ephem.sun
    print(f"Sun position: {sun_skycoord[0].cartesian.xyz}")
    print(f"Observer location: {sun_skycoord.frame.obsgeoloc[0]}")
    
    # Get Moon position as seen from spacecraft
    moon_skycoord = ephem.moon
    print(f"Moon position: {moon_skycoord[0].cartesian.xyz}")
    
    # Get Earth position relative to spacecraft
    earth_skycoord = ephem.earth
    print(f"Earth position: {earth_skycoord[0].cartesian.xyz}")
    # Note: Earth position is the negative of spacecraft GCRS position

**Ground Observatory Example**

.. code-block:: python

    # Create ground observatory ephemeris
    obs = rust_ephem.GroundEphemeris(
        latitude=19.8207,    # Mauna Kea
        longitude=-155.468,
        height=4205.0,
        begin=begin,
        end=end,
        step_size=60
    )
    
    # Access observatory and celestial positions
    obs_gcrs = obs.gcrs      # Observatory in GCRS frame
    obs_itrs = obs.itrs      # Observatory in ITRS frame
    sun_from_obs = obs.sun   # Sun as seen from observatory
    moon_from_obs = obs.moon # Moon as seen from observatory
    
    # Calculate Sun altitude/azimuth
    sun_altaz = sun_from_obs.transform_to(
        astropy.coordinates.AltAz(
            obstime=sun_from_obs.obstime,
            location=astropy.coordinates.EarthLocation.from_geocentric(
                *obs_itrs[0].cartesian.xyz
            )
        )
    )
    print(f"Sun altitude: {sun_altaz[0].alt}")
    print(f"Sun azimuth: {sun_altaz[0].az}")

**SPICE Ephemeris Example**

.. code-block:: python

    # Create SPICE ephemeris
    spice_ephem = rust_ephem.SPICEEphemeris(
        spk_path="spacecraft.bsp",
        naif_id=-999,
        begin=begin,
        end=end,
        step_size=60
    )
    
    # Access all frames as SkyCoord
    gcrs = spice_ephem.gcrs
    itrs = spice_ephem.itrs
    sun = spice_ephem.sun
    moon = spice_ephem.moon
    earth = spice_ephem.earth

The SkyCoord objects include:

- Full position and velocity information
- Proper coordinate frame metadata
- Observer location (obsgeoloc) and velocity (obsgeovel) for celestial bodies
- Compatible with astropy's transformation framework