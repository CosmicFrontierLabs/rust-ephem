"""``import rust_ephem`` must not pay for packages only some features use."""

import subprocess
import sys
from pathlib import Path


def test_import_does_not_load_astroquery_or_astropy(tmp_path: Path) -> None:
    code = (
        "import sys, rust_ephem; "
        "print(sorted(m for m in ('astroquery', 'astropy', 'pyvo', 'duckdb', 'pyarrow')"
        " if m in sys.modules))"
    )
    # Run from outside the repository: with ``-c`` the working directory is on
    # sys.path, and the source tree has no compiled extension when the package
    # was installed from a wheel.
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_vizier_is_still_reachable_for_downloads() -> None:
    from astroquery.vizier import Vizier  # type: ignore[import-untyped]

    from rust_ephem import bright_stars

    assert bright_stars.Vizier is Vizier
