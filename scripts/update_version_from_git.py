#!/usr/bin/env python3
"""
Update Cargo.toml version based on git tags.

This script is intended for local builds and mirrors what the CI does in
.github/workflows/build-wheels.yml (where tags are used to update the Cargo.toml
version before the build). It does not commit changes; it's a local convenience.

Usage:
    scripts/update_version_from_git.py [--dry-run]

If HEAD is exactly on a tag, uses that version (e.g., v0.1.13 -> 0.1.13).
If HEAD is past a tag, generates a dev version (e.g., v0.1.13-5-gabcdef -> 0.1.13-dev.5).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parents[1] / "Cargo.toml"


def get_version_from_git() -> str | None:
    """Return a version string based on git describe.

    Returns:
        - "X.Y.Z" if exactly on a tag vX.Y.Z
        - "X.Y.Z-dev.N" if N commits past tag vX.Y.Z
        - None if no tags found or git not available
    """
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags"], stderr=subprocess.DEVNULL
        )
        desc = out.decode("utf-8").strip()

        # Strip leading 'v' if present
        if desc.startswith("v"):
            desc = desc[1:]

        # Check if it's exactly a tag (no commits past) or has dev commits
        # Format: X.Y.Z or X.Y.Z-N-gHASH
        match = re.match(r"^(\d+\.\d+\.\d+)(?:-(\d+)-g[a-f0-9]+)?$", desc)
        if match:
            base_version = match.group(1)
            commits_past = match.group(2)
            if commits_past:
                # Dev version: X.Y.Z-dev.N (Rust semver compatible)
                return f"{base_version}-dev.{commits_past}"
            else:
                # Exact tag
                return base_version

        # Fallback: return as-is if pattern doesn't match
        return desc

    except subprocess.CalledProcessError:
        # No tags or git not available
        return None


def update_cargo_toml(version: str, dry_run: bool = False) -> bool:
    """Update Cargo.toml's version to the provided version.

    Returns True when a change was made, False otherwise.
    """
    text = CARGO_TOML.read_text(encoding="utf-8")
    # Replace the first occurrence of a line like 'version = "x.y.z"'
    version_pat = re.compile(r'^(version\s*=\s*")[^"]+("\s*)$', re.MULTILINE)

    def repl(m: "re.Match[str]") -> str:
        return f"{m.group(1)}{version}{m.group(2)}"

    new_text, count = version_pat.subn(repl, text, count=1)
    if count == 0:
        print("Warning: Could not find version line in Cargo.toml", file=sys.stderr)
        return False

    if new_text == text:
        # No change needed - silent success for pre-commit
        return False

    if dry_run:
        print("Dry-run: would update Cargo.toml version to", version)
    else:
        CARGO_TOML.write_text(new_text, encoding="utf-8")
        print(f"Updated Cargo.toml version to {version}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report version to be used without changing files",
    )
    args = parser.parse_args()

    version = get_version_from_git()
    if not version:
        # No tags - silent success for pre-commit
        sys.exit(0)

    updated = update_cargo_toml(version, dry_run=args.dry_run)
    if updated:
        print(f"Updated Cargo.toml version to {version}")
