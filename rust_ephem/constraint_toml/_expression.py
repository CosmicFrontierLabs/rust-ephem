"""Boolean-expression parser and AST -> constraint-config resolution.

Parses the ``&``/``|``/``^``/``~``/``at_least(...)`` expression strings
produced by :mod:`rust_ephem.constraint_toml._writer` back into a constraint
config tree. Deliberately hand-rolled (no ``eval``) so a malformed or
malicious expression string can only ever raise :class:`ValueError`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tomlkit

from ..constraints import CombinedConstraintConfig, ConstraintConfig

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_RE = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z0-9_]*)|([&|^~(),]))")

# Expression AST node: ("name", str) | ("not", node) | ("and"|"or"|"xor", [node, ...])
# | ("at_least", int, [node, ...])
_AstNode = tuple[Any, ...]


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
