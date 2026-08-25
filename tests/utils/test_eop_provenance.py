from __future__ import annotations

import rust_ephem


def test_eop_provenance_reports_each_provider() -> None:
    provenance = rust_ephem.get_eop_provenance()

    assert set(provenance) == {"ut1", "polar_motion"}
    for provider in (provenance["ut1"], provenance["polar_motion"]):
        assert isinstance(provider["available"], bool)
        if not provider["available"]:
            assert provider == {"available": False}
            continue

        assert provider["source_url"].startswith("https://")
        assert len(provider["sha256"]) == 64
        assert all(char in "0123456789abcdef" for char in provider["sha256"])
        assert provider["loaded_from"] in {
            "download",
            "fresh_cache",
            "stale_cache",
        }
        assert provider["stale"] is (provider["loaded_from"] == "stale_cache")
