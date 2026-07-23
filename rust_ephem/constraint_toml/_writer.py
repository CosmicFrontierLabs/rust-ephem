"""Constraint tree -> annotated TOML document compiler.

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

from ..constraints import (
    AndConstraint,
    AtLeastConstraint,
    BoresightOffsetConstraint,
    NotConstraint,
    OrConstraint,
    RustConstraintMixin,
    XorConstraint,
)
from ._fields import _add_comment_lines, _write_leaf_fields

# Operator precedence, matching Python's own bitwise operator precedence
# exactly (~ tightest, then &, then ^, then | loosest) since these are the
# same symbols RustConstraintMixin overloads for combinator construction.
_PRECEDENCE_OR = 0
_PRECEDENCE_XOR = 1
_PRECEDENCE_AND = 2
_PRECEDENCE_NOT = 3
_PRECEDENCE_ATOM = 4

_LOGICAL_OPERATORS: dict[str, tuple[str, int]] = {
    "and": ("&", _PRECEDENCE_AND),
    "or": ("|", _PRECEDENCE_OR),
    "xor": ("^", _PRECEDENCE_XOR),
}

_EXPRESSION_LEGEND = (
    "Boolean expression combining the [definitions.*] below by name. "
    "Operators follow Python's own precedence: ~ (NOT) binds tightest, then "
    "& (AND), then ^ (XOR), then | (OR); use parentheses to override. "
    "at_least(k, a, b, ...) spells out a k-of-n threshold combinator."
)

_CONSTRAINT_EXPRESSION_COMMENT = (
    "constraint_expression -- boolean expression for the wrapped constraint, "
    "see [definitions.*] {ref}"
)


class _Compiler:
    """Flattens a constraint tree into named, unnested leaf definitions plus a
    boolean expression string that references them by name."""

    def __init__(self) -> None:
        # name -> (model, constraint_expression or None)
        self.definitions: dict[str, tuple[RustConstraintMixin, str | None]] = {}

    def _unique_name(self, base: str) -> str:
        if base not in self.definitions:
            return base
        index = 2
        while f"{base}_{index}" in self.definitions:
            index += 1
        return f"{base}_{index}"

    def compile(self, node: RustConstraintMixin) -> tuple[str, int]:
        if isinstance(node, (AndConstraint, OrConstraint, XorConstraint)):
            symbol, level = _LOGICAL_OPERATORS[node.type]
            parts = []
            for child in node.constraints:
                text, prec = self.compile(child)
                parts.append(f"({text})" if prec < level else text)
            return f" {symbol} ".join(parts), level

        if isinstance(node, NotConstraint):
            text, prec = self.compile(node.constraint)
            text = f"({text})" if prec < _PRECEDENCE_NOT else text
            return f"~{text}", _PRECEDENCE_NOT

        if isinstance(node, AtLeastConstraint):
            parts = [self.compile(child)[0] for child in node.constraints]
            args = ", ".join([str(node.min_violated), *parts])
            return f"at_least({args})", _PRECEDENCE_ATOM

        # A concrete constraint: BoresightOffsetConstraint carries its own
        # scalar config plus one nested constraint, so its inner tree still
        # needs compiling; every other leaf type is fully self-contained.
        constraint_expression = None
        if isinstance(node, BoresightOffsetConstraint):
            constraint_expression, _ = self.compile(node.constraint)
        name = self._unique_name(node.type)  # type: ignore[attr-defined]
        self.definitions[name] = (node, constraint_expression)
        return name, _PRECEDENCE_ATOM


def _is_combinator(node: RustConstraintMixin) -> bool:
    return isinstance(
        node,
        (AndConstraint, OrConstraint, XorConstraint, NotConstraint, AtLeastConstraint),
    )


def _write_constraint_expression(
    tbl: Any, constraint_expression: str, ref: str
) -> None:
    _add_comment_lines(tbl, _CONSTRAINT_EXPRESSION_COMMENT.format(ref=ref))
    tbl.add("constraint_expression", constraint_expression)
    tbl.add(tomlkit.nl())


def _write_definitions(
    definitions: dict[str, tuple[RustConstraintMixin, str | None]], doc: Any
) -> None:
    if not definitions:
        return
    table = tomlkit.table(is_super_table=True)
    for name, (model, constraint_expression) in definitions.items():
        sub = tomlkit.table()
        _write_leaf_fields(model, sub)
        if constraint_expression is not None:
            _write_constraint_expression(sub, constraint_expression, "above")
        table.add(name, sub)
    doc.append("definitions", table)


def constraint_to_toml_document(
    constraint: RustConstraintMixin,
) -> tomlkit.TOMLDocument:
    """Render a constraint config as an annotated :class:`tomlkit.TOMLDocument`."""
    doc = tomlkit.document()

    if _is_combinator(constraint):
        compiler = _Compiler()
        expression, _ = compiler.compile(constraint)
        _add_comment_lines(doc, _EXPRESSION_LEGEND)
        doc.add("expression", expression)
        doc.add(tomlkit.nl())
        _write_definitions(compiler.definitions, doc)
        return doc

    if isinstance(constraint, BoresightOffsetConstraint):
        compiler = _Compiler()
        constraint_expression, _ = compiler.compile(constraint.constraint)
        _write_leaf_fields(constraint, doc)
        _write_constraint_expression(doc, constraint_expression, "below")
        _write_definitions(compiler.definitions, doc)
        return doc

    _write_leaf_fields(constraint, doc)
    return doc


def constraint_to_toml_string(constraint: RustConstraintMixin) -> str:
    """Render a constraint config as an annotated TOML string."""
    return tomlkit.dumps(constraint_to_toml_document(constraint))


def write_constraint_toml(constraint: RustConstraintMixin, path: str | Path) -> None:
    """Write a constraint config to *path* as annotated TOML."""
    Path(path).write_text(constraint_to_toml_string(constraint), encoding="utf-8")
