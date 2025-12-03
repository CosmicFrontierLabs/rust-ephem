# Building the Documentation

This directory contains the Sphinx documentation for ``rust-ephem``.

## Quick Start (Local Build)

```bash
# From repository root
cd docs

# Install dependencies
pip install -r requirements-docs.txt

# Build HTML docs
make html

# View in browser
open _build/html/index.html
```

## Full Build with Extension

For complete documentation including API autodoc:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install build tools and extension
pip install maturin
maturin develop --release

# Install Sphinx requirements
pip install -r docs/requirements-docs.txt

# Build docs
cd docs
make html
```

## ReadTheDocs

The documentation is configured for automatic builds on ReadTheDocs. See
``.readthedocs.yaml`` in the repository root for configuration.

## Notes

- If the native extension is not installed, Sphinx will mock the module
  (API signatures won't be available, but other docs will build)
- See ``conf.py`` for Sphinx configuration and mock settings
- Run ``make clean`` before rebuilding if you see stale content
