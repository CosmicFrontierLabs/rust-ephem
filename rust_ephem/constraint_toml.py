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
"""

from __future__ import annotations

import re
import textwrap
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Any

import annotated_types
import tomlkit
from pydantic.fields import FieldInfo

from .constraints import (
    AndConstraint,
    AtLeastConstraint,
    BoresightOffsetConstraint,
    CombinedConstraintConfig,
    ConstraintConfig,
    NotConstraint,
    OrConstraint,
    RustConstraintMixin,
    XorConstraint,
)

_COMMENT_WIDTH = 88

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

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_RE = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z0-9_]*)|([&|^~(),]))")

# Expression AST node: ("name", str) | ("not", node) | ("and"|"or"|"xor", [node, ...])
# | ("at_least", int, [node, ...])
_AstNode = tuple[Any, ...]


# ---------------------------------------------------------------------------
# Field-comment annotation helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(annotation: Any) -> Any:
    """Strip an ``Optional[X]``/``X | None`` wrapper down to ``X``."""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_constraint_field(field_info: FieldInfo) -> bool:
    """True for a field whose value is itself a nested constraint config
    (e.g. ``BoresightOffsetConstraint.constraint``), which is serialized
    separately as a ``constraint_expression``/``[definitions.*]`` pair
    rather than written inline as a scalar."""
    annotation = field_info.annotation
    origin = typing.get_origin(annotation)
    args = (
        typing.get_args(annotation)
        if origin in (typing.Union, types.UnionType)
        else (annotation,)
    )
    return bool(args) and all(
        isinstance(arg, type) and issubclass(arg, RustConstraintMixin) for arg in args
    )


def _bounds_note(field_info: FieldInfo) -> str | None:
    notes: list[str] = []
    for constraint in field_info.metadata:
        if isinstance(constraint, annotated_types.Ge):
            notes.append(f">= {constraint.ge}")
        elif isinstance(constraint, annotated_types.Gt):
            notes.append(f"> {constraint.gt}")
        elif isinstance(constraint, annotated_types.Le):
            notes.append(f"<= {constraint.le}")
        elif isinstance(constraint, annotated_types.Lt):
            notes.append(f"< {constraint.lt}")
        elif isinstance(constraint, annotated_types.MinLen):
            notes.append(f"min length {constraint.min_length}")
        elif isinstance(constraint, annotated_types.MaxLen):
            notes.append(f"max length {constraint.max_length}")
    return ", ".join(notes) if notes else None


def _choices_note(annotation: Any) -> str | None:
    ann = _unwrap_optional(annotation)
    if typing.get_origin(ann) is typing.Literal:
        return "choices: " + ", ".join(str(a) for a in typing.get_args(ann))
    if isinstance(ann, type) and issubclass(ann, Enum):
        return "choices: " + ", ".join(str(m.value) for m in ann)
    return None


def _default_repr(default: Any) -> Any:
    if isinstance(default, Enum):
        return default.value
    if isinstance(default, tuple):
        return list(default)
    return default


def _field_comment(field_info: FieldInfo) -> str:
    parts: list[str] = []
    if field_info.description:
        parts.append(field_info.description)
    bounds = _bounds_note(field_info)
    if bounds:
        parts.append(f"({bounds})")
    choices = _choices_note(field_info.annotation)
    if choices:
        parts.append(f"({choices})")
    if not field_info.is_required() and field_info.default is not None:
        parts.append(f"[default: {_default_repr(field_info.default)!r}]")
    return " ".join(parts)


def _class_summary(model_cls: type[RustConstraintMixin]) -> list[str]:
    """Return the docstring summary (before any ``Attributes:`` section) as
    a list of paragraphs, each with its wrapped lines already joined onto
    one logical line."""
    doc = model_cls.__doc__ or ""
    summary, _, _ = doc.partition("Attributes:")
    paragraphs = [
        " ".join(line.strip() for line in para.splitlines() if line.strip())
        for para in summary.strip().split("\n\n")
    ]
    return [para for para in paragraphs if para]


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_normalize_scalar(v) for v in value]
    return value


def _add_comment_lines(tbl: Any, text: str) -> None:
    for line in textwrap.wrap(text, _COMMENT_WIDTH) or [text]:
        tbl.add(tomlkit.comment(line))


def _write_field(tbl: Any, name: str, field_info: FieldInfo, value: Any) -> None:
    comment_body = _field_comment(field_info)
    if value is None:
        _add_comment_lines(tbl, f"{name} (not set) -- {comment_body}")
    else:
        _add_comment_lines(tbl, f"{name} -- {comment_body}")
        tbl.add(name, _normalize_scalar(value))
    tbl.add(tomlkit.nl())


def _write_leaf_fields(model: RustConstraintMixin, tbl: Any) -> None:
    """Write a concrete (non-combinator) constraint's docstring, ``type``, and
    scalar fields. A nested-constraint field (e.g. ``constraint`` on
    ``BoresightOffsetConstraint``) is skipped -- the caller writes it
    separately as ``constraint_expression``."""
    paragraphs = _class_summary(type(model))
    for paragraph in paragraphs:
        _add_comment_lines(tbl, paragraph)
    if paragraphs:
        tbl.add(tomlkit.nl())

    tbl.add("type", model.type)  # type: ignore[attr-defined]
    tbl.add(tomlkit.nl())

    for name, field_info in type(model).model_fields.items():
        if name == "type" or _is_constraint_field(field_info):
            continue
        _write_field(tbl, name, field_info, getattr(model, name))


# ---------------------------------------------------------------------------
# Tree -> (definitions, expression) compiler
# ---------------------------------------------------------------------------


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
            _add_comment_lines(
                sub,
                "constraint_expression -- boolean expression for the wrapped constraint, see [definitions.*] above",
            )
            sub.add("constraint_expression", constraint_expression)
            sub.add(tomlkit.nl())
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
        paragraphs = _class_summary(type(constraint))
        for paragraph in paragraphs:
            _add_comment_lines(doc, paragraph)
        if paragraphs:
            doc.add(tomlkit.nl())
        doc.add("type", constraint.type)
        doc.add(tomlkit.nl())
        for name, field_info in type(constraint).model_fields.items():
            if name == "type" or _is_constraint_field(field_info):
                continue
            _write_field(doc, name, field_info, getattr(constraint, name))
        _add_comment_lines(
            doc,
            "constraint_expression -- boolean expression for the wrapped constraint, see [definitions.*] below",
        )
        doc.add("constraint_expression", constraint_expression)
        doc.add(tomlkit.nl())
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


# ---------------------------------------------------------------------------
# Expression parser + tree reconstruction
# ---------------------------------------------------------------------------


class _ExpressionSyntaxError(ValueError):
    pass


class _ExpressionParser:
    """Recursive-descent parser for the ``&``/``|``/``^``/``~``/``at_least(...)``
    expression grammar, mirroring Python's own bitwise operator precedence:

        expr    := or_expr
        or_expr := xor_expr ('|' xor_expr)*
        xor_expr:= and_expr ('^' and_expr)*
        and_expr:= not_expr ('&' not_expr)*
        not_expr:= '~' not_expr | primary
        primary := IDENT | 'at_least' '(' INT ',' expr (',' expr)* ')' | '(' expr ')'
    """

    def __init__(self, text: str) -> None:
        self._tokens = self._tokenize(text)
        self._pos = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            match = _TOKEN_RE.match(text, pos)
            if match is None or match.end() == pos:
                if text[pos:].strip() == "":
                    break
                raise _ExpressionSyntaxError(
                    f"Unexpected character {text[pos : pos + 1]!r} in constraint expression: {text!r}"
                )
            pos = match.end()
            token = next(g for g in match.groups() if g is not None)
            tokens.append(token)
        return tokens

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        token = self._peek()
        if token is None:
            raise _ExpressionSyntaxError("Unexpected end of constraint expression")
        self._pos += 1
        return token

    def _expect(self, token: str) -> None:
        if self._advance() != token:
            raise _ExpressionSyntaxError(f"Expected {token!r} in constraint expression")

    def parse(self) -> _AstNode:
        node = self._or_expr()
        if self._peek() is not None:
            raise _ExpressionSyntaxError(
                f"Unexpected token {self._peek()!r} in constraint expression"
            )
        return node

    def _binary(self, symbol: str, kind: str, next_level: Any) -> _AstNode:
        parts = [next_level()]
        while self._peek() == symbol:
            self._advance()
            parts.append(next_level())
        return parts[0] if len(parts) == 1 else (kind, parts)

    def _or_expr(self) -> _AstNode:
        return self._binary("|", "or", self._xor_expr)

    def _xor_expr(self) -> _AstNode:
        return self._binary("^", "xor", self._and_expr)

    def _and_expr(self) -> _AstNode:
        return self._binary("&", "and", self._not_expr)

    def _not_expr(self) -> _AstNode:
        if self._peek() == "~":
            self._advance()
            return ("not", self._not_expr())
        return self._primary()

    def _primary(self) -> _AstNode:
        token = self._peek()
        if token is None:
            raise _ExpressionSyntaxError("Unexpected end of constraint expression")
        if token == "(":
            self._advance()
            node = self._or_expr()
            self._expect(")")
            return node
        if _IDENT_RE.fullmatch(token):
            self._advance()
            if token == "at_least":
                self._expect("(")
                count_token = self._advance()
                if not count_token.isdigit():
                    raise _ExpressionSyntaxError(
                        "at_least(...) expects an integer threshold as its first argument"
                    )
                self._expect(",")
                children = [self._or_expr()]
                while self._peek() == ",":
                    self._advance()
                    children.append(self._or_expr())
                self._expect(")")
                return ("at_least", int(count_token), children)
            return ("name", token)
        raise _ExpressionSyntaxError(
            f"Unexpected token {token!r} in constraint expression"
        )


def _resolve_ast(
    node: _AstNode,
    definitions: dict[str, Any],
    resolving: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    kind = node[0]
    if kind == "name":
        name = node[1]
        if name not in definitions:
            raise _ExpressionSyntaxError(
                f"Unknown constraint reference {name!r} in constraint_expression"
            )
        if name in resolving:
            raise _ExpressionSyntaxError(
                f"Circular constraint_expression reference: {name!r}"
            )
        entry = dict(definitions[name])
        constraint_expression = entry.pop("constraint_expression", None)
        if constraint_expression is not None:
            inner_ast = _ExpressionParser(constraint_expression).parse()
            entry["constraint"] = _resolve_ast(
                inner_ast, definitions, resolving | {name}
            )
        return entry
    if kind == "not":
        return {
            "type": "not",
            "constraint": _resolve_ast(node[1], definitions, resolving),
        }
    if kind in ("and", "or", "xor"):
        return {
            "type": kind,
            "constraints": [
                _resolve_ast(child, definitions, resolving) for child in node[1]
            ],
        }
    if kind == "at_least":
        _, min_violated, children = node
        return {
            "type": "at_least",
            "min_violated": min_violated,
            "constraints": [
                _resolve_ast(child, definitions, resolving) for child in children
            ],
        }
    raise _ExpressionSyntaxError(
        f"Unknown expression node: {node!r}"
    )  # pragma: no cover


def parse_constraint_toml(text: str) -> ConstraintConfig:
    """Parse a TOML string (as produced by :func:`constraint_to_toml_string`) back
    into the matching :data:`~rust_ephem.constraints.ConstraintConfig` model."""
    parsed = tomlkit.parse(text)

    if "expression" in parsed:
        definitions = parsed.get("definitions", {})
        ast = _ExpressionParser(parsed["expression"]).parse()
        return CombinedConstraintConfig.validate_python(_resolve_ast(ast, definitions))

    if "constraint_expression" in parsed:
        definitions = parsed.get("definitions", {})
        root = {
            k: v
            for k, v in parsed.items()
            if k not in ("definitions", "constraint_expression")
        }
        ast = _ExpressionParser(parsed["constraint_expression"]).parse()
        root["constraint"] = _resolve_ast(ast, definitions)
        return CombinedConstraintConfig.validate_python(root)

    return CombinedConstraintConfig.validate_python(parsed)


def load_constraint_toml(path: str | Path) -> ConstraintConfig:
    """Read a constraint config TOML file from *path*."""
    return parse_constraint_toml(Path(path).read_text(encoding="utf-8"))
