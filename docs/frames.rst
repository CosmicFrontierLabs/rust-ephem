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
   - Convention: Vallado/SGP4 TEME, using IAU 1982 mean sidereal time
   - GCRS conversion: Consistent with the TEME → ITRS → GCRS route below

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
2. **TEME → ITRS** using SOFA ``gmst82`` and optional polar motion, without
   the TIO locator ``s'`` (the Vallado TEME convention)
3. **TEME → GCRS** using the rotation derived from that TEME → ITRS transform
   and the inverse IAU 2006/2000A GCRS → ITRS transform. The direct and composed
   routes agree in both position and velocity when given the same EOP inputs
   and polar-motion setting.

With ``T = W(xp, yp, 0) * R3(GMST82)`` and the celestial-to-terrestrial matrix
``M`` below, the GCRS → TEME matrix is ``T^T * M``. Polar motion cancels, leaving
``R3(s') * R3(ERA - GMST82) * Q``. This avoids both an extra transformation
through ITRS and replacing the SGP4 mean-equinox convention with GMST06.
TEME conversions include the matrix-rate velocity term; rotating velocity
alone omits motion of the reference frame.

For ground-based ephemeris:

1. **Geodetic → ITRS** using WGS84 ellipsoid
2. **ITRS → GCRS** using the inverse celestial-to-terrestrial transformation below

For GCRS ↔ ITRS conversions (including file, OEM, SPICE, Parquet, and ground
ephemerides), the forward matrix is ``M = W * R3(ERA) * Q``:

- ``Q``: frame bias, IAU 2006 precession, and IAU 2000A nutation, evaluated in TT
- ``R3(ERA)``: Earth rotation angle evaluated in UT1
- ``W``: polar motion and the TIO locator ``s'``

The implementation uses the SOFA ``c2t06a`` decomposition through ``sofars``.
For nearby GCRS ↔ ITRS and TEME ↔ GCRS samples, the slowly varying ``Q``
matrix is interpolated cubically on a fixed 300-second TT grid with a four-node,
batch-local cache. Its derivative comes from the same cubic, without additional
series evaluations or stencil-driven cache eviction. Isolated samples use two
full series evaluations at TT ±0.5 seconds to obtain a midpoint matrix and its
centered derivative. Cache selection is local: a large gap does not disable
interpolation elsewhere in the batch. Interpolated and isolated rotations are
orthonormalized, with the same operation differentiated for velocity consistency.
TT and UT1 retain separate whole-day and fractional-day values throughout;
sub-microsecond offsets are not rounded into a large Julian date first.
Earth rotation and polar motion are evaluated at each requested time. No global
cache or cached Earth-orientation data is introduced. Setting
``polar_motion=False`` sets ``xp = yp = 0``; it does **not** disable precession,
nutation, frame bias, or ``s'``.

Positions transform as ``r_itrs = M * r_gcrs`` and velocities as
``v_itrs = M * v_gcrs + Mdot * r_gcrs``. The inverse uses the transpose of ``M``
and subtracts the same frame-motion term. The same rule is used for TEME
transforms. ``Mdot`` combines analytic ERA, GMST82 and ``s'`` rates with the
celestial derivative using the product rule. The celestial derivative is
retained: omitting it can exceed the velocity tolerance even in low Earth orbit.
Earth-orientation parameters are held fixed
locally: polar-motion rates and length-of-day corrections are not modeled.
Observed celestial-pole offsets (``dX``, ``dY``) are also not applied.
The interpolation retains the position/velocity regression tolerances of 1 mm
and 1 micrometre/s for the tested near-Earth and beyond-geosynchronous states;
these are numerical agreement tolerances, not absolute orbit-accuracy claims.
Round trips are additionally checked at lunar, 1 AU and 30 AU distances with
floating-point-scaled tolerances. Inputs must have exactly one six-component
state per timestamp; invalid shapes raise ``ValueError``, including identity
conversions.

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
