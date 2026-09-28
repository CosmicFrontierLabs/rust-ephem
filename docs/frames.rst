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
2. **TEME → ITRS** using GMST and optional polar motion
3. **TEME → GCRS** separately, using the equation of equinoxes and the inverse
   bias-precession-nutation matrix

For ground-based ephemeris:

1. **Geodetic → ITRS** using WGS84 ellipsoid
2. **ITRS → GCRS** using the inverse celestial-to-terrestrial transformation below

For GCRS ↔ ITRS conversions (including file, OEM, SPICE, Parquet, and ground
ephemerides), the forward matrix is ``M = W * R3(ERA) * Q``:

- ``Q``: frame bias, IAU 2006 precession, and IAU 2000A nutation, evaluated in TT
- ``R3(ERA)``: Earth rotation angle evaluated in UT1
- ``W``: polar motion and the TIO locator ``s'``

The implementation uses SOFA's ``c2t06a`` through ``sofars``. Setting
``polar_motion=False`` sets ``xp = yp = 0``; it does **not** disable precession,
nutation, frame bias, or ``s'``.

Positions transform as ``r_itrs = M * r_gcrs`` and velocities as
``v_itrs = M * v_gcrs + Mdot * r_gcrs``. The inverse uses the transpose of ``M``
and subtracts the same frame-motion term. ``Mdot`` is evaluated by a centered
one-second difference in TT and UT1. Earth-orientation parameters are held fixed
locally: polar-motion rates and length-of-day corrections are not modeled.
Observed celestial-pole offsets (``dX``, ``dY``) are also not applied.

Implementation Details
----------------------

- **sofars library**: Pure-Rust SOFA routines for astronomical transformations
- **IAU 2006/2000A model**: Precession-nutation matrix
- **Frame bias**: Proper ICRS/GCRS alignment
- **Polar motion**: Optional correction for Earth axis movement
- **UT1 corrections**: Account for Earth's irregular rotation

Accuracy Impact
---------------

Frame transformation accuracy depends on the input frame, epoch, distance from
Earth's center, and available Earth-orientation data. Precession and nutation
are always included in GCRS ↔ ITRS conversions. UT1 and polar-motion accuracy
remain limited by provider coverage and fallback behavior; enabling polar
motion does not guarantee accurate data outside the provider's date range.

Enable high-accuracy mode:

.. code-block:: python

   rust_ephem.init_ut1_provider()
   rust_ephem.init_eop_provider()

   ephem = rust_ephem.TLEEphemeris(..., polar_motion=True)

See Also
--------

- :doc:`time_systems` — Time scale handling affects frame transformations
- :doc:`accuracy_precision` — Detailed accuracy information
