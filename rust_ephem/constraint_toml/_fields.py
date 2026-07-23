"""Field-comment annotation helpers shared by the TOML writer.

Every constraint config field already carries a Pydantic ``Field(description=...)``
plus validation bounds (``ge``/``le``/``min_length``/...). This module turns that
metadata into human-readable comments above each key.
"""

from __future__ import annotations

import textwrap
import types
import typing
from enum import Enum
from typing import Any

import annotated_types
import tomlkit
from pydantic.fields import FieldInfo

from ..constraints import RustConstraintMixin

_COMMENT_WIDTH = 88


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


def _normalize_scalar(value: Any) -> Any:
    """Convert a field value (or Pydantic default) into a plain, TOML/repr-friendly
    scalar: enums become their ``.value``, tuples become lists, recursively."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_normalize_scalar(v) for v in value]
    return value


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
        parts.append(f"[default: {_normalize_scalar(field_info.default)!r}]")
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
