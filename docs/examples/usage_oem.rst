Using OEMEphemeris
===================

Example showing how to construct an OEMEphemeris and access geodetic
latitude/longitude and height for each timestamp.

.. code-block:: python

    import datetime as dt
    import rust_ephem as re

    # Create a small OEM ephemeris from the provided test file
    begin = dt.datetime(2032, 6, 28, 10, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2032, 7, 5, 10, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds (1-minute resolution)

    eph = re.OEMEphemeris(
        oem_path="AXIS27June2032MJ2000_200km_long.oem",
        begin=begin,
        end=end,
        step_size=step_size,
        polar_motion=False,
    )

    # Interpolated timestamps
    print(f"Number of timestamps: {len(eph.timestamp)}")

    # Access geodetic arrays derived from ITRS positions
    print(f"Latitude (deg) at first timestamp: {eph.latitude_deg[0]}")
    print(f"Longitude (deg) at first timestamp: {eph.longitude_deg[0]}")
    print(f"Height (m) at first timestamp: {eph.height_m[0]}")

    # Use the height (meters) directly instead of computing altitude
    # from GCRS vector norm (which includes Earth center offset)
    print(f"Height as Quantity: {eph.height[0]}")

    # Access GCRS position/velocity
    gcrs_pv = eph.gcrs_pv
    print("GCRS position (km):", gcrs_pv.position[0])

    # Index nearest a particular time
    idx = eph.index(dt.datetime(2032, 7, 1, 12, 0, 0, tzinfo=dt.timezone.utc))
    print("Index closest to target time:", idx)

    # Constraint example: avoid pointing directly at Sun for first timestamp
    sun_constraint = re.Constraint.sun_proximity(min_angle=45.0)
    satisfied = sun_constraint.in_constraint(
        time=eph.timestamp[idx],
        ephemeris=eph,
        target_ra=180.0,
        target_dec=30.0,
    )
    print("Constraint satisfied at index?", satisfied)
