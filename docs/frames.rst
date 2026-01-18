Coordinate Frames
=================

``rust-ephem`` works with several standard astronomical coordinate frames.
Understanding these frames is essential for correct ephemeris usage.

Reference Frames
----------------

**TEME (True Equator, Mean Equinox)**
   The output frame from the SGP4 propagator. Based on the true equator and
   mean equinox of date. Used internally for TLE propagation.

   - Origin: Earth center
   - Reference: True equator, mean equinox of date
   - Use case: TLE propagation (SGP4 native output)
   - GCRS conversion: Applies the equation of equinoxes to align mean vs true equinox

**ITRS (International Terrestrial Reference System)**
   An Earth-fixed coordinate system that rotates with the Earth. Useful for
   ground-based applications and geographic calculations.

   - Origin: Earth center
   - Reference: Rotates with Earth's crust
   - Use case: Ground station positions, geographic coordinates

**GCRS (Geocentric Celestial Reference System)**
   A modern celestial reference frame aligned with ICRS but centered at Earth.
   The preferred frame for most astronomical calculations.

   - Origin: Earth center
   - Reference: ICRS (quasi-inertial, does not rotate with Earth)
   - Use case: Celestial observations, spacecraft tracking

Frame Properties
----------------

All ephemeris classes provide coordinates in multiple frames:

.. code-block:: python

   import rust_ephem

   ephem = rust_ephem.TLEEphemeris(...)

   # Position/velocity data (PositionVelocityData objects)
   ephem.teme_pv   # TEME frame (TLEEphemeris only)
   ephem.itrs_pv   # ITRS frame
   ephem.gcrs_pv   # GCRS frame

   # Astropy SkyCoord objects
   ephem.itrs      # ITRS SkyCoord
   ephem.gcrs      # GCRS SkyCoord

Transformation Pipeline
-----------------------

For TLE-based ephemeris:

1. **SGP4 Propagation** → TEME position/velocity
2. **TEME → ITRS** using Earth rotation, precession-nutation
3. **ITRS → GCRS** using polar motion, frame bias

For ground-based ephemeris:

1. **Geodetic → ITRS** using WGS84 ellipsoid
2. **ITRS → GCRS** using Earth rotation, polar motion

Implementation Details
----------------------

- **ERFA library**: IAU-standard routines for astronomical transformations
- **IAU 2006 model**: Modern precession-nutation matrix
- **Frame bias**: Proper ICRS/GCRS alignment
- **Polar motion**: Optional correction for Earth axis movement
- **UT1 corrections**: Account for Earth's irregular rotation

Accuracy Impact
---------------

Frame transformation accuracy depends on available corrections:

==============================  ============
Configuration                   Accuracy
==============================  ============
Default (no corrections)        ~100 meters
UT1 corrections enabled         ~20 meters
UT1 + polar motion              ~10-20 meters
==============================  ============

Enable high-accuracy mode:

.. code-block:: python

   rust_ephem.init_ut1_provider()
   rust_ephem.init_eop_provider()

   ephem = rust_ephem.TLEEphemeris(..., polar_motion=True)

See Also
--------

- :doc:`time_systems` — Time scale handling affects frame transformations
- :doc:`accuracy_precision` — Detailed accuracy information
