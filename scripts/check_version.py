#!/usr/bin/env python3
"""
Pre-commit hook to verify Cargo.toml version matches the git tag.

Exit codes:
    0 - Version matches or no tags found (success)
    1 - Version mismatch (failure)

Usage:
    scripts/check_version.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parents[1] / "Cargo.toml"


def get_version_from_git() -> str | None:
    """Return a version string based on git describe."""
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags"], stderr=subprocess.DEVNULL
        )
        desc = out.decode("utf-8").strip()

        if desc.startswith("v"):
            desc = desc[1:]

        match = re.match(r"^(\d+\.\d+\.\d+)(?:-(\d+)-g[a-f0-9]+)?$", desc)
        if match:
            base_version = match.group(1)
            commits_past = match.group(2)
            if commits_past:
                return f"{base_version}-dev.{commits_past}"
            else:
                return base_version

        return desc

    except subprocess.CalledProcessError:
        return None


def get_cargo_version() -> str | None:
    """Read current version from Cargo.toml."""
    text = CARGO_TOML.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def main() -> int:
    git_version = get_version_from_git()
    if not git_version:
        print("No git tags found, skipping version check")
        return 0

    cargo_version = get_cargo_version()
    if not cargo_version:
        print("Could not read version from Cargo.toml", file=sys.stderr)
        return 1

    if git_version == cargo_version:
        print(f"✓ Version {cargo_version} matches git tag")
        return 0
    else:
        print(
            f"✗ Version mismatch: Cargo.toml has {cargo_version}, git tag indicates {git_version}",
            file=sys.stderr,
        )
        print("  Run: python scripts/update_version_from_git.py", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
