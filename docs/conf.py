# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add project root to sys.path so autodoc can import the package when installed
sys.path.insert(0, os.path.abspath(".."))

project = "rust-ephem"
author = "Jamie A. Kennea"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

# If the extension module isn't available at docs build time, mock it so the
# docs can still build on systems that haven't built the native extension.
autodoc_mock_imports = []

# Use the manually-committed stub pages in _autosummary/ instead of regenerating
autosummary_generate = False

# Suppress mocked object warnings since we intentionally mock rust_ephem
# Also suppress autosummary stub warnings since we're not generating stubs
suppress_warnings = [
    "autodoc.mocked_object",
    "autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ----------------------------------------------

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except Exception:
    html_theme = "alabaster"
html_static_path = ["_static"]

# Autodoc default options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
