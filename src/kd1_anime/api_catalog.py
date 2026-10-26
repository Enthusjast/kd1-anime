"""Exact API lookup independent of semantic document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from kd1_anime.manim_catalog_0201 import CATALOG


@dataclass(frozen=True, slots=True)
class ApiSymbol:
    name: str
    status: Literal["found", "missing", "unknown"]
    version: str
    signature: str | None = None
    bases: tuple[str, ...] = ()


def installed_version() -> str:
    try:
        return version("manim")
    except PackageNotFoundError:
        return ""


def lookup_symbol(name: str, manim_version: str | None = None) -> ApiSymbol:
    selected = installed_version() if manim_version is None else manim_version
    if selected != CATALOG["version"]:
        return ApiSymbol(name, "unknown", selected)
    record = CATALOG["symbols"].get(name)
    if record is None:
        return ApiSymbol(name, "missing", selected)
    return ApiSymbol(name, "found", selected, record["signature"], tuple(record.get("bases", ())))
