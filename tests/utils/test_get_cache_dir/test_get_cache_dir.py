from pathlib import Path

from rust_ephem import get_cache_dir


class TestGetCacheDir:
    def test_get_cache_dir_is_str(self) -> None:
        cache_dir = get_cache_dir()
        assert isinstance(cache_dir, str)

    def test_get_cache_dir_exists(self) -> None:
        cache_dir = get_cache_dir()
        assert Path(cache_dir).exists()

    def test_get_cache_dir_is_dir(self) -> None:
        cache_dir = get_cache_dir()
        assert Path(cache_dir).is_dir()
