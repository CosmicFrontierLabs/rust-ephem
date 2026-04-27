Using FileEphemeris
===================

``FileEphemeris`` reads pre-computed spacecraft state vectors from a plain-text
file and resamples them to a uniform output grid using Hermite interpolation —
the same approach used by :class:`~rust_ephem.OEMEphemeris` for CCSDS OEM files.

It is designed to handle the common case where a mission simulator (e.g. STK,
GMAT, ODTBX) exports an ephemeris file and you want to feed it into the
``rust-ephem`` constraint-evaluation pipeline without re-propagating from scratch.

Supported file formats
----------------------

The parser is deliberately liberal.  It expects:

* Lines with at least 7 whitespace-separated tokens where columns 2–7 are all
  floating-point numbers: ``<time>  <x>  <y>  <z>  <vx>  <vy>  <vz>``
* Lines starting with ``#``, ``//``, or ``!`` are treated as comments and
  skipped.
* Blank lines are skipped.
* Header key-value pairs are detected automatically; the keys listed below are
  recognised regardless of capitalisation.

**STK .e files** are a primary target.  A minimal STK header looks like:

.. code-block:: text

    stk.v.12.0

    BEGIN Ephemeris

        NumberOfEphemerisPoints   1728001
        ScenarioEpoch             15 Oct 2028 00:00:00.000000
        CoordinateSystem          J2000

        EphemerisTimePosVel

     0.0000000000000000e+00  -1.3728675769e+07  -8.4374672301e+07  ...
     1.0000000000000000e+00  -1.3726287021e+07  -8.4373514108e+07  ...
     ...

    END Ephemeris

Other file layouts (simple CSVs with a comment header, custom simulator output,
etc.) are also handled as long as the data rows match the 7-column pattern.

Timestamp formats
-----------------

The time column is parsed according to the ``time_format`` parameter:

+----------------+------------------------------------------------------------+
| ``"auto"``     | Try numeric offset first, then ISO 8601 / STK date strings.|
| (default)      |                                                            |
+----------------+------------------------------------------------------------+
| ``"seconds"``  | Offset from a reference epoch in seconds.                  |
+----------------+------------------------------------------------------------+
| ``"days"``     | Offset from a reference epoch in days.                     |
+----------------+------------------------------------------------------------+
| ``"iso8601"``  | Absolute UTC datetime: ``YYYY-MM-DDTHH:MM:SS.ffffff``.     |
+----------------+------------------------------------------------------------+

For offset-based formats the reference epoch is taken from the first recognised
epoch key in the file header (``ScenarioEpoch``, ``Epoch``, ``T0``, …) or from
the ``epoch`` constructor parameter.

Supported epoch string formats:

* STK natural-language: ``15 Oct 2028 00:00:00.000000``
* ISO 8601 with T: ``2028-10-15T00:00:00.000000``
* ISO 8601 with space: ``2028-10-15 00:00:00.000000``
* RFC 3339 (with timezone offset): ``2028-10-15T00:00:00Z``
* Date only: ``15 Oct 2028`` (time defaults to 00:00:00)

Input coordinate frames
-----------------------

The coordinate frame is read from the file header (key ``CoordinateSystem`` or
``ref_frame``) and can be overridden with the ``frame`` constructor parameter.

**GCRS-compatible frames** (J2000, EME2000, GCRF, GCRS, ICRF, ICRF2, ICRF3)
  Data are treated as inertial and stored directly in GCRS.  ITRS is derived by
  applying the ERA rotation (and optionally polar motion).

**Earth-fixed frames** (ITRS, ECEF, ECF, FIXED, TERRESTRIAL)
  Data are treated as Earth-fixed and stored as ITRS.  GCRS is derived by
  applying the inverse ERA rotation.

Units
-----

Defaults are **km** for position and **km/s** for velocity, matching most
simulation output.  Override with ``position_unit`` and ``velocity_unit`` if
your file uses different units.

Supported position units: ``"km"`` (default), ``"m"``, ``"cm"``.

Supported velocity units: ``"km/s"`` (default), ``"m/s"``, ``"cm/s"``.

Basic example (STK .e file)
---------------------------

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    begin = datetime(2028, 10, 15, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(2028, 10, 15, 1, 0, 0, tzinfo=timezone.utc)

    eph = re.FileEphemeris(
        "LazuliSat_Ephemeris_1sec_20281015to20281104.e",
        begin=begin,
        end=end,
        step_size=60,        # resample to 1-minute output grid
    )

    print(f"Source frame : {eph.source_frame}")          # J2000
    print(f"File rows    : {eph.file_pv.position.shape[0]}")
    print(f"Output rows  : {eph.gcrs_pv.position.shape[0]}")
    print(f"Position[0]  : {eph.gcrs_pv.position[0]} km")
    print(f"Latitude[0]  : {eph.latitude_deg[0]:.4f} deg")

Explicit epoch override
-----------------------

If the file has no header epoch (e.g. a stripped CSV), supply it explicitly:

.. code-block:: python

    from datetime import datetime, timezone

    eph = re.FileEphemeris(
        "trajectory_offsets.txt",
        begin=begin,
        end=end,
        epoch=datetime(2028, 10, 15, 0, 0, 0, tzinfo=timezone.utc),
        time_format="seconds",
    )

ISO 8601 absolute timestamps
-----------------------------

If your file uses absolute timestamps in each row:

.. code-block:: python

    # File looks like:
    # 2028-10-15T00:00:00  -13728675.8  -84374672.3  -13426109.5  ...
    # 2028-10-15T00:01:00  ...

    eph = re.FileEphemeris(
        "trajectory_iso.txt",
        begin=begin,
        end=end,
        time_format="iso8601",
    )

Unit conversion
---------------

If your simulator exports positions in metres:

.. code-block:: python

    eph = re.FileEphemeris(
        "trajectory_meters.e",
        begin=begin,
        end=end,
        position_unit="m",
        velocity_unit="m/s",
    )

    # All standard properties (gcrs_pv, latitude_deg, …) are in km / km/s.
    print(eph.source_position_unit)   # "m"   (as supplied)
    print(eph.gcrs_pv.position_unit)  # "km"  (internal representation)

Earth-fixed input frame
-----------------------

If the file is in an Earth-fixed (ECEF/ITRS) frame:

.. code-block:: python

    eph = re.FileEphemeris(
        "trajectory_ecef.txt",
        begin=begin,
        end=end,
        frame="ECEF",
    )

Inspecting raw file data
------------------------

The raw (uninterpolated) state vectors and timestamps are accessible via
``file_pv`` and ``file_timestamp``:

.. code-block:: python

    raw = eph.file_pv
    print(f"File has {raw.position.shape[0]} state vectors")

    ts = eph.file_timestamp
    print(f"First file timestamp : {ts[0]}")
    print(f"Last  file timestamp : {ts[-1]}")

Constraint evaluation
---------------------

``FileEphemeris`` participates in the full constraint-evaluation pipeline:

.. code-block:: python

    re.ensure_planetary_ephemeris()

    constraint = re.SunConstraint(min_angle=45.0) | re.MoonConstraint(min_angle=10.0)
    result = constraint.evaluate(eph, target_ra=83.63, target_dec=22.01)

    print(f"Visibility windows: {len(result.visibility)}")
    for window in result.visibility:
        print(f"  {window.start_time} → {window.end_time}")

Type checking
-------------

``FileEphemeris`` is a registered virtual subclass of :class:`~rust_ephem.Ephemeris`:

.. code-block:: python

    assert isinstance(eph, re.Ephemeris)   # True
