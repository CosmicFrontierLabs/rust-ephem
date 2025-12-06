#!/usr/bin/env python3
"""
Update Cargo.toml version to the current git tag if one exists.

This script is intended for local builds and mirrors what the CI does in
.github/workflows/build-wheels.yml (where tags are used to update the Cargo.toml
version before the build). It does not commit changes; it's a local convenience.

Usage:
    scripts/update_version_from_git.py [--dry-run]

It will try to discover a git tag that points to HEAD. If none is present, the
script prints a message and does nothing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parents[1] / "Cargo.toml"


def get_tag() -> str | None:
    """Return the tag pointing at HEAD, or None if not tagged."""
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"], stderr=subprocess.DEVNULL
        )
        tag = out.decode("utf-8").strip()
        return tag
    except subprocess.CalledProcessError:
        # Not on a tag or git not available
        return None


def transform_tag_to_version(tag: str) -> str:
    """Transform a tag string into a semver-ish version if possible.

    Common pattern is vX.Y.Z or X.Y.Z; strip leading 'v' if present.
    """
    if tag.startswith("v"):
        return tag[1:]
    return tag


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
        print(f"Cargo.toml already at version {version}")
        return False

    if dry_run:
        print("Dry-run: would update Cargo.toml version to", version)
    else:
        CARGO_TOML.write_text(new_text, encoding="utf-8")
        print("Updated Cargo.toml version to", version)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report version to be used without changing files",
    )
    args = parser.parse_args()

    tag = get_tag()
    if not tag:
        print("No git tag pointing at HEAD found, not updating Cargo.toml")
        sys.exit(0)

    version = transform_tag_to_version(tag)
    updated = update_cargo_toml(version, dry_run=args.dry_run)
    if updated:
        print("Success — Cargo.toml changed to", version)
    else:
        print("No change required")
