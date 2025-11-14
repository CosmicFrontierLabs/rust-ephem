Coordinate frames
=================

``rust-ephem`` works with three main coordinate frames:

- TEME (True Equator, Mean Equinox): direct output from SGP4
- ITRS (International Terrestrial Reference System): Earth-fixed frame
- GCRS (Geocentric Celestial Reference System): modern celestial reference frame

Transformation pipeline
-----------------------

1. Propagate a TLE using SGP4 to obtain position-velocity in TEME
2. Transform TEME → ITRS using Earth rotation, precession-nutation, and polar motion
3. Transform ITRS → GCRS (or directly TEME → GCRS where appropriate)

Implementation notes
--------------------

- ERFA routines (IAU standards) are used for astronomical transformations.
- The precession-nutation matrix follows the IAU 2006 model.
- Position-velocity vectors are represented as 3D positions (km) and velocities (km/s).
- Matrix operations are optimized to avoid unnecessary allocations.

See also: :doc:`time_systems` for time-scale details that affect frame conversion.
