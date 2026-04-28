"""Tests for the get_bright_stars catalog utility (VizieR mocked)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import rust_ephem
from rust_ephem import get_bright_stars


def _make_table(ras: list[float], decs: list[float], vmags: list[float]) -> MagicMock:
    """Build a fake astropy Table-like object whose items are proper numpy arrays."""
    data = {
        "_RA.icrs": np.array(ras, dtype=float),
        "_DE.icrs": np.array(decs, dtype=float),
        "Vmag": np.array(vmags, dtype=float),
    }
    table = MagicMock()
    table.__getitem__ = MagicMock(side_effect=lambda key: data[key])
    return table


def _make_vizier_result(table: MagicMock) -> MagicMock:
    result = MagicMock()
    result.__len__ = MagicMock(return_value=1)
    result.__bool__ = MagicMock(return_value=True)
    result.__getitem__ = MagicMock(return_value=table)
    return result


# ── Cache behaviour ────────────────────────────────────────────────────────────


class TestGetBrightStarsCaching:
    """VizieR is mocked; tests verify cache read/write semantics."""

    def test_first_call_downloads_and_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        table = _make_table([10.0, 20.0], [5.0, -5.0], [4.0, 5.0])
        vizier_result = _make_vizier_result(table)

        with patch("rust_ephem.bright_stars.Vizier") as MockVizier:
            instance = MockVizier.return_value
            instance.query_constraints.return_value = vizier_result

            stars = get_bright_stars(mag_limit=6.0, refresh=True)

        assert len(stars) == 2
        assert all(len(s) == 2 for s in stars)
        # Cache file should exist
        cache_files = list(tmp_path.glob("hipparcos_vmag_*.npy"))
        assert len(cache_files) == 1

    def test_second_call_uses_cache_no_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        # Pre-populate cache
        data = np.array([[10.0, 5.0, 4.0], [20.0, -5.0, 5.5]])
        np.save(tmp_path / "hipparcos_vmag_6.00.npy", data)

        with patch("rust_ephem.bright_stars.Vizier") as MockVizier:
            stars = get_bright_stars(mag_limit=6.0)
            MockVizier.assert_not_called()

        assert len(stars) == 2

    def test_magnitude_filter_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        # Cache has 3 stars: two bright (4, 5) and one dim (7)
        data = np.array([[10.0, 5.0, 4.0], [20.0, -5.0, 5.0], [30.0, 0.0, 7.0]])
        np.save(tmp_path / "hipparcos_vmag_8.00.npy", data)

        stars = get_bright_stars(mag_limit=6.0)
        assert len(stars) == 2  # only the two with vmag <= 6.0

    def test_broader_cache_serves_tighter_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        # A broad cache (mag_limit=8.0) exists; request mag_limit=5.0
        data = np.array([[10.0, 5.0, 4.0], [20.0, -5.0, 7.5]])
        np.save(tmp_path / "hipparcos_vmag_8.00.npy", data)

        with patch("rust_ephem.bright_stars.Vizier") as MockVizier:
            stars = get_bright_stars(mag_limit=5.0)
            MockVizier.assert_not_called()

        assert len(stars) == 1  # only star with vmag=4.0 passes the 5.0 limit

    def test_refresh_ignores_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        # Put a cache file with different content
        old_data = np.array([[99.0, 99.0, 3.0]])
        np.save(tmp_path / "hipparcos_vmag_6.00.npy", old_data)

        fresh_table = _make_table([1.0, 2.0], [0.0, 0.0], [4.0, 5.0])
        vizier_result = _make_vizier_result(fresh_table)

        with patch("rust_ephem.bright_stars.Vizier") as MockVizier:
            instance = MockVizier.return_value
            instance.query_constraints.return_value = vizier_result

            stars = get_bright_stars(mag_limit=6.0, refresh=True)

        # Should get the fresh data (2 stars), not the cached (1 star)
        assert len(stars) == 2

    def test_cache_mag_limit_larger_than_mag_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        table = _make_table([10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [4.0, 6.0, 7.5])
        vizier_result = _make_vizier_result(table)

        with patch("rust_ephem.bright_stars.Vizier") as MockVizier:
            instance = MockVizier.return_value
            instance.query_constraints.return_value = vizier_result

            # Download up to 8.0, return only up to 5.0
            stars = get_bright_stars(mag_limit=5.0, cache_mag_limit=8.0, refresh=True)

        assert len(stars) == 1  # only vmag=4.0
        # Cache written at the broader limit
        assert (tmp_path / "hipparcos_vmag_8.00.npy").exists()

    def test_cache_mag_limit_less_than_mag_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_mag_limit"):
            get_bright_stars(mag_limit=7.0, cache_mag_limit=5.0)


# ── Return format ──────────────────────────────────────────────────────────────


class TestGetBrightStarsReturnFormat:
    def test_returns_list_of_tuples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        data = np.array([[15.5, -3.2, 4.1], [270.0, 45.0, 5.8]])
        np.save(tmp_path / "hipparcos_vmag_7.00.npy", data)

        stars = get_bright_stars(mag_limit=7.0)
        assert isinstance(stars, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in stars)

    def test_ra_dec_values_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        data = np.array([[15.5, -3.2, 4.1]])
        np.save(tmp_path / "hipparcos_vmag_7.00.npy", data)

        stars = get_bright_stars(mag_limit=7.0)
        ra, dec = stars[0]
        assert ra == pytest.approx(15.5)
        assert dec == pytest.approx(-3.2)

    def test_stars_suitable_for_constraint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stars returned by get_bright_stars can be passed directly to Constraint.bright_star."""
        monkeypatch.setattr("rust_ephem.bright_stars._cache_dir", lambda: tmp_path)

        data = np.array([[10.0, 5.0, 4.5], [20.0, -5.0, 6.0]])
        np.save(tmp_path / "hipparcos_vmag_7.00.npy", data)

        stars = get_bright_stars(mag_limit=7.0)
        c = rust_ephem.Constraint.bright_star(stars=stars, fov_radius=1.0)
        assert c is not None
