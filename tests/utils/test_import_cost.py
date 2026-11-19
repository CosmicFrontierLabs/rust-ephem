"""``import rust_ephem`` must not pay for packages only some features use."""

import subprocess
import sys


def test_import_does_not_load_astroquery_or_astropy() -> None:
    code = (
        "import sys, rust_ephem; "
        "print(sorted(m for m in ('astroquery', 'astropy', 'pyvo', 'duckdb', 'pyarrow')"
        " if m in sys.modules))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_vizier_is_still_reachable_for_downloads() -> None:
    from astroquery.vizier import Vizier  # type: ignore[import-untyped]

    import rust_ephem.bright_stars as bright_stars

    assert bright_stars.Vizier is Vizier
