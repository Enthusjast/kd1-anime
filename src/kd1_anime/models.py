"""Validated copying at domain and service boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def plain_value(value: Any) -> Any:
    """Break nested model identities so Pydantic cannot skip their validation."""
    if isinstance(value, BaseModel):
        return plain_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain_value(item) for item in value)
    return value


def validated_copy(model: M, *, update: dict[str, Any] | None = None, deep: bool = False) -> M:
    values = plain_value(model)
    values.update(plain_value(update or {}))
    return type(model).model_validate(deepcopy(values) if deep else values)


class ValidatedModel(BaseModel):
    """Keep the familiar copy API, with validation when values are changed."""

    def model_copy(self, *, update=None, deep=False):
        if update is None:
            return super().model_copy(deep=deep)
        return validated_copy(self, update=update, deep=deep)


def snapshot_value(value):
    """Validate owned model copies while retaining the service-facing types."""
    if isinstance(value, BaseModel):
        return validated_copy(value, deep=True)
    if isinstance(value, dict):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot_value(item) for item in value)
    return deepcopy(value)
