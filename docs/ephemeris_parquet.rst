Using ParquetEphemeris
======================

``ParquetEphemeris`` reads pre-computed spacecraft state vectors from a
`Parquet <https://parquet.apache.org/>`_ file via
`DuckDB <https://duckdb.org/>`_, then resamples them to a uniform output grid
using Hermite interpolation — the same approach used by
:class:`~rust_ephem.OEMEphemeris` and :class:`~rust_ephem.FileEphemeris`.

It is designed for the increasingly common case where mission ephemerides
are produced as columnar Parquet datasets, often stored on S3 or another
object store, and need to feed the ``rust-ephem`` constraint-evaluation
pipeline directly without an intermediate text export.

Supported sources
-----------------

The Parquet ``source`` argument is forwarded to DuckDB's
``read_parquet()``, so any URI that DuckDB understands works:

* **Local files**: ``"path/to/file.parquet"`` or globs like
  ``"data/sat_*.parquet"`` (DuckDB unions matching files automatically).
* **Amazon S3**: ``"s3://bucket/key.parquet"``.
* **DigitalOcean Spaces** and other S3-compatibles: any ``s3://``
  URI plus the ``s3_endpoint`` constructor kwarg
  (e.g. ``"nyc3.digitaloceanspaces.com"``).
* **Google Cloud Storage**: ``"gcs://bucket/key.parquet"``.
* **Cloudflare R2**: ``"r2://bucket/key.parquet"``.
* **HTTP(S)**: ``"https://host/path.parquet"`` for public buckets and CDNs.

Required schema
---------------

The Parquet file must contain at minimum **seven columns**:

* a timestamp column,
* three position columns (``x``, ``y``, ``z``),
* three velocity columns (``vx``, ``vy``, ``vz``).

The defaults are ``time`` for the timestamp and ``x/y/z`` + ``vx/vy/vz`` for
the state vectors. Override via the ``time_col``, ``pos_cols`` and
``vel_cols`` constructor kwargs.

The timestamp column must be castable by DuckDB to ``TIMESTAMP`` /
``TIMESTAMPTZ``. A column written by Arrow/pandas with timezone-aware
``datetime64`` values, or an ISO 8601 string column, both work out of the
box. Other columns in the file are ignored.

Authentication
--------------

Cloud access uses DuckDB's built-in
``credential_chain`` provider, which transparently picks up the standard
AWS environment variables. None of the credentials ever pass through
``rust-ephem``:

============================== ==================================================
Variable                       Purpose
============================== ==================================================
``AWS_ACCESS_KEY_ID``          Access key (S3 or S3-compatible)
``AWS_SECRET_ACCESS_KEY``      Secret key
``AWS_SESSION_TOKEN``          Session token (for STS / temporary creds)
``AWS_REGION`` /
``AWS_DEFAULT_REGION``         Region (override with ``s3_region`` kwarg)
============================== ==================================================

For DigitalOcean Spaces or any non-AWS S3-compatible service, additionally
pass the host as ``s3_endpoint`` (without ``https://``):

.. code-block:: python

    ParquetEphemeris(
        "s3://my-spaces-bucket/sat42.parquet",
        begin, end, step_size=60,
        s3_endpoint="nyc3.digitaloceanspaces.com",
    )

Time-range pre-filtering
------------------------

To keep memory and bandwidth down on large historical archives,
``ParquetEphemeris`` issues a ``WHERE`` clause on the DuckDB query that
restricts loaded rows to ``[begin − 1 h, end + 1 h]``. The 1 hour margin
ensures Hermite interpolation has neighbours on either side of the
requested window. Combined with Parquet's row-group statistics this gives
DuckDB enough information to skip irrelevant chunks of large files.

If a Parquet file's sampling rate is sparser than 1 minute, or if the
requested range sits very close to the file's earliest / latest sample,
you may need to pad the source data so the margin is satisfied.

Optional dependency
-------------------

DuckDB is an optional dependency. Install with:

.. code-block:: bash

    pip install rust-ephem[parquet]

or simply:

.. code-block:: bash

    pip install duckdb

Constructing one
----------------

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    eph = re.ParquetEphemeris(
        "spacecraft_states.parquet",
        begin=begin,
        end=end,
        step_size=60,           # resample to 1-minute output grid
    )

    print(f"Source frame : {eph.source_frame}")          # "GCRS" by default
    print(f"Source rows  : {eph.file_pv.position.shape[0]}")
    print(f"Output rows  : {eph.gcrs_pv.position.shape[0]}")
    print(f"Position[0]  : {eph.gcrs_pv.position[0]} km")
    print(f"Latitude[0]  : {eph.latitude_deg[0]:.4f} deg")

Custom column names
-------------------

If your schema uses non-default names:

.. code-block:: python

    eph = re.ParquetEphemeris(
        "ephemeris.parquet",
        begin=begin, end=end, step_size=60,
        time_col="epoch_utc",
        pos_cols=("rx", "ry", "rz"),
        vel_cols=("rdotx", "rdoty", "rdotz"),
    )

Column names may contain only ASCII letters, digits, and underscores
(``[A-Za-z0-9_]+``). This includes names that begin with a digit. Names
with spaces or punctuation are rejected to prevent SQL injection — re-name
them in your data or query before ingestion.

Reading from S3
---------------

.. code-block:: bash

    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_REGION=us-east-1

.. code-block:: python

    eph = re.ParquetEphemeris(
        "s3://my-bucket/ephemerides/sat42.parquet",
        begin=begin, end=end, step_size=60,
    )

Reading from DigitalOcean Spaces
--------------------------------

.. code-block:: bash

    # Spaces uses the same env-var convention as AWS S3
    export AWS_ACCESS_KEY_ID="DO00..."
    export AWS_SECRET_ACCESS_KEY="..."

.. code-block:: python

    eph = re.ParquetEphemeris(
        "s3://my-space/ephemerides/sat42.parquet",
        begin=begin, end=end, step_size=60,
        s3_endpoint="nyc3.digitaloceanspaces.com",
    )

Filtering by satellite ID
-------------------------

If a single Parquet file contains states for multiple spacecraft, narrow
the result with ``where_clause``:

.. code-block:: python

    eph = re.ParquetEphemeris(
        "s3://fleet/states.parquet",
        begin=begin, end=end, step_size=60,
        where_clause="sat_id = 42",
    )

The clause is ANDed onto the time-range filter, so DuckDB still gets to
push it down to Parquet row groups.

Unit overrides
--------------

If your Parquet stores positions in metres:

.. code-block:: python

    eph = re.ParquetEphemeris(
        "states_meters.parquet",
        begin=begin, end=end,
        position_unit="m",
        velocity_unit="m/s",
    )

    # All standard properties (gcrs_pv, latitude_deg, …) are still in km / km/s.
    print(eph.source_position_unit)   # "m" (as supplied)
    print(eph.gcrs_pv.position_unit)  # "km" (internal representation)

Earth-fixed (ITRS/ECEF) input
-----------------------------

If the data is in an Earth-fixed (ITRS/ECEF) frame, set ``frame``:

.. code-block:: python

    eph = re.ParquetEphemeris(
        "states_ecef.parquet",
        begin=begin, end=end,
        frame="ECEF",
    )

Inspecting raw data
-------------------

The raw, uninterpolated state vectors and timestamps are exposed via
``file_pv`` and ``file_timestamp``:

.. code-block:: python

    raw = eph.file_pv
    print(f"Pulled {raw.position.shape[0]} state vectors from Parquet")

    ts = eph.file_timestamp
    print(f"Earliest sample : {ts[0]}")
    print(f"Latest   sample : {ts[-1]}")

Evaluating constraints
----------------------

``ParquetEphemeris`` participates in the full constraint-evaluation
pipeline:

.. code-block:: python

    re.ensure_planetary_ephemeris()

    constraint = re.SunConstraint(min_angle=45.0) | re.MoonConstraint(min_angle=10.0)
    result = constraint.evaluate(eph, target_ra=83.63, target_dec=22.01)

    print(f"Visibility windows: {len(result.visibility)}")
    for window in result.visibility:
        print(f"  {window.start_time} → {window.end_time}")

Ephemeris ABC subclass
----------------------

``ParquetEphemeris`` is a registered virtual subclass of
:class:`~rust_ephem.Ephemeris`:

.. code-block:: python

    assert isinstance(eph, re.Ephemeris)   # True
