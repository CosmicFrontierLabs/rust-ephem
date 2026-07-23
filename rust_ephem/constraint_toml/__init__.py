"""
Annotated TOML export/import for constraint configuration models.

Every constraint config field already carries a Pydantic ``Field(description=...)``
plus validation bounds (``ge``/``le``/``min_length``/...). This module turns that
metadata into human-readable comments above each key, so a constraint written out
with :func:`constraint_to_toml_string` is a self-documenting, hand-editable
template rather than a bare data dump.

Boolean combinators (``AndConstraint``, ``OrConstraint``, ``XorConstraint``,
``NotConstraint``, ``AtLeastConstraint``) are never written as nested TOML
tables -- nesting a nontrivial boolean tree into TOML's dotted-path table
headers produces an unreadable wall of same-looking headers with no visible
structure. Instead, every concrete (non-combinator) constraint is written as
its own flat, unnested ``[definitions.<name>]`` table, and the boolean tree
that combines them collapses into a single ``expression`` string such as
``"sun | moon | ~eclipse"``. The operators mirror the ``&``/``|``/``^``/``~``
overloads already defined on :class:`~rust_ephem.constraints.RustConstraintMixin`,
including Python's own operator precedence (``~`` tightest, then ``&``, then
``^``, then ``|``); ``at_least(k, a, b, ...)`` spells out the k-of-n threshold
combinator. A constraint tree with no combinators at all (a single leaf, or a
single :class:`~rust_ephem.constraints.BoresightOffsetConstraint`) is written
with no ``definitions``/``expression`` indirection at all.

This package is organized as:

- :mod:`._fields` -- Pydantic field metadata -> TOML comment rendering.
- :mod:`._writer` -- constraint tree -> annotated TOML document compiler.
- :mod:`._expression` -- boolean-expression parser and tree reconstruction.
"""

from __future__ import annotations

from ._expression import load_constraint_toml, parse_constraint_toml
from ._writer import (
    constraint_to_toml_document,
    constraint_to_toml_string,
    write_constraint_toml,
)

__all__ = [
    "constraint_to_toml_document",
    "constraint_to_toml_string",
    "write_constraint_toml",
    "parse_constraint_toml",
    "load_constraint_toml",
]
