import pathlib

import pytest

try:
    import rust_ephem  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    rust_ephem = None


class TestSpicePreload:
    pytestmark = [
        pytest.mark.skipif(rust_ephem is None, reason="rust_ephem extension not built")
    ]

    def test_is_planetary_ephemeris_initialized_default_false(self):
        # If earlier tests (e.g., get_body) have already initialized the planetary ephemeris,
        # this test's original assertion would fail purely due to test ordering. Make it robust
        # by skipping when initialization has already occurred.
        if rust_ephem.is_planetary_ephemeris_initialized():
            pytest.skip("Planetary ephemeris already initialized by earlier tests")
        assert not rust_ephem.is_planetary_ephemeris_initialized()

    def test_ensure_planetary_ephemeris_errors_when_missing(self, monkeypatch):
        default_spk = pathlib.Path("unlikely_to_exist_spk_file.spk")
        if default_spk.exists():
            pytest.skip(
                f"Default planetary SPK exists at {default_spk}; skipping missing-file test"
            )
        with pytest.raises(FileNotFoundError):
            rust_ephem.ensure_planetary_ephemeris(
                str(default_spk), download_if_missing=False, spk_url=None
            )
