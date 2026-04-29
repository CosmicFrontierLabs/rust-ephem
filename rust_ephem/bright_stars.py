"""Utilities for fetching and caching bright star catalogs.

Stars are sourced from the Hipparcos catalog (ESA 1997) via VizieR using
astroquery.  The full table for a given magnitude limit is saved to disk as a
numpy array so subsequent calls are instant without network access.

Typical usage::

    from rust_ephem import get_bright_stars, Constraint

    stars = get_bright_stars(mag_limit=7.0)
    c = Constraint.bright_star(
        stars=stars,
        fov_polygon=[(-0.25, -0.15), (0.25, -0.15), (0.25, 0.15), (-0.25, 0.15)],
    )
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astroquery.vizier import Vizier  # type: ignore[import-untyped]
from numpy.typing import NDArray

__all__ = ["get_bright_stars"]

_CATALOG_ID = "I/239/hip_main"
_COLUMNS = ["_RA.icrs", "_DE.icrs", "Vmag"]
_CACHE_PREFIX = "hipparcos_vmag_"


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _cache_dir() -> Path:
    import rust_ephem

    return Path(rust_ephem.get_cache_dir())


def _cache_path(mag_limit: float) -> Path:
    return _cache_dir() / f"{_CACHE_PREFIX}{mag_limit:.2f}.npy"


def _find_usable_cache(mag_limit: float) -> Path | None:
    """Return the path of the smallest cached file whose magnitude limit covers mag_limit.

    A cache file at limit L covers a request for limit M when L >= M, because
    all stars brighter than M are a subset of stars brighter than L.
    We prefer the smallest qualifying L to minimise the cost of the in-memory
    filter step.
    """
    pattern = str(_cache_dir() / f"{_CACHE_PREFIX}*.npy")
    candidates: list[tuple[float, Path]] = []
    for p in glob.glob(pattern):
        stem = Path(p).stem
        if not stem.startswith(_CACHE_PREFIX):
            continue
        try:
            limit = float(stem[len(_CACHE_PREFIX) :])
        except ValueError:
            continue
        if limit >= mag_limit:
            candidates.append((limit, Path(p)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ── Download ───────────────────────────────────────────────────────────────────


def _download(cache_mag_limit: float) -> NDArray[np.float64]:
    """Fetch Hipparcos stars brighter than *cache_mag_limit* from VizieR.

    Returns an (N, 3) float64 array with columns [ra_deg, dec_deg, vmag].
    """

    v = Vizier(columns=_COLUMNS, row_limit=-1)
    result = v.query_constraints(
        catalog=_CATALOG_ID,
        Vmag=f"<{cache_mag_limit}",
    )

    if not result or len(result) == 0:
        raise ValueError(
            f"VizieR returned no stars for Vmag < {cache_mag_limit}. "
            "Check your network connection and that astroquery can reach CDS."
        )

    table = result[0]

    # astropy Table columns may be MaskedColumns; convert safely to plain float arrays.
    ra = np.asarray(table["_RA.icrs"], dtype=float)
    dec = np.asarray(table["_DE.icrs"], dtype=float)
    vmag = np.asarray(table["Vmag"], dtype=float)

    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vmag)
    return np.column_stack([ra[valid], dec[valid], vmag[valid]])


# ── Public API ─────────────────────────────────────────────────────────────────


def get_bright_stars(
    mag_limit: float = 7.0,
    cache_mag_limit: float | None = None,
    *,
    refresh: bool = False,
) -> list[tuple[float, float]]:
    """Return Hipparcos bright stars suitable for use with ``Constraint.bright_star``.

    Stars brighter than *mag_limit* (Johnson V magnitude) are returned as
    ``(ra_deg, dec_deg)`` pairs in ICRS / J2000 coordinates.

    The catalog is downloaded from VizieR on the first call and cached to disk
    so subsequent calls are instant.  A cache file covers all stars up to a
    given magnitude; requesting a tighter limit from an existing broader cache
    never triggers a download.

    Parameters
    ----------
    mag_limit:
        Return only stars brighter than this V magnitude (lower = brighter).
        Default ``7.0`` gives roughly 4 000 stars.
    cache_mag_limit:
        Magnitude limit used when writing the on-disk cache.  Defaults to
        *mag_limit*.  Set this larger than *mag_limit* to download a broader
        dataset that can serve future calls with tighter limits without another
        network round-trip.  For example::

            # Downloads all stars brighter than 8.0, caches them, then
            # returns only those brighter than 6.0.
            stars = get_bright_stars(6.0, cache_mag_limit=8.0)

    refresh:
        If ``True``, ignore any existing on-disk cache and re-download from
        VizieR, then update the cache.  Default ``False``.

    Returns
    -------
    list[tuple[float, float]]
        ``(ra_deg, dec_deg)`` pairs for every star brighter than *mag_limit*.

    Raises
    ------
    ImportError
        If *astroquery* is not installed and no suitable cache exists.
    ValueError
        If *cache_mag_limit* < *mag_limit*, or VizieR returns no rows.
    """
    if cache_mag_limit is None:
        cache_mag_limit = mag_limit
    if cache_mag_limit < mag_limit:
        raise ValueError(
            f"cache_mag_limit ({cache_mag_limit}) must be >= mag_limit ({mag_limit})"
        )

    cache_path: Path | None = None if refresh else _find_usable_cache(mag_limit)

    if cache_path is None:
        data = _download(cache_mag_limit)
        dest = _cache_path(cache_mag_limit)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, data)
        cache_path = dest

    data = np.load(cache_path)

    # Filter to the actually requested magnitude limit
    mask = data[:, 2] <= mag_limit
    filtered = data[mask]

    return [(float(row[0]), float(row[1])) for row in filtered]
