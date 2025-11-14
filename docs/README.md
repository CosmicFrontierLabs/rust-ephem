# Building the documentation

This project exposes a Python extension module named ``rust_ephem`` built from
Rust sources using maturin. The Sphinx documentation in this directory uses
``autodoc`` to import the compiled extension. There are two common workflows:

1. ReadTheDocs (recommended for hosted builds)
   - Configure RTD to run `pip install .` or use the pyproject.toml build with
     maturin. RTD will then build and import the extension during the docs
     build.

2. Local build (developer machine)
   - Install the extension in your Python environment so Sphinx can import it.

Local build steps (example)

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install build tooling
pip install -U pip build maturin

# Build and install the Python extension in editable/develop mode
# maturin develop --release

# Install Sphinx and requirements for docs
pip install -r docs/requirements.txt

# Build the docs
cd docs
sphinx-build -b html . _build/html
```

Notes
-----
- If you cannot or do not want to build the native extension locally, the
  Sphinx config mocks the extension module so the docs can still be generated
  (but member signatures and docstrings coming from the compiled extension
  will not be available). See `conf.py` for the mocked module list.
- For ReadTheDocs, configure the project to run the necessary build commands
  (for example: `pip install .`). See ReadTheDocs documentation for native
  extensions and Rust-based wheels.
