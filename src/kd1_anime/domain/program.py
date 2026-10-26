"""Versioned, explicit parameters for the opt-in deterministic 2D compiler."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, FiniteFloat, model_validator

from kd1_anime.models import ValidatedModel as BaseModel

Vector = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Positive = Annotated[FiniteFloat, Field(gt=0, le=1000)]


class ObjectBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
    position: Vector
    color_key: Literal["primary", "secondary", "highlight", "warning", "foreground"] = "primary"
    initially_active: bool = False
    exported: bool = False


class TextObject(ObjectBase):
    kind: Literal["Text", "Tex", "MathTex"]
    text: str = Field(min_length=1, max_length=4000)
    font_size: Positive = 32


class CircleObject(ObjectBase):
    kind: Literal["Circle"]
    radius: Positive


class RectangleObject(ObjectBase):
    kind: Literal["Rectangle"]
    width: Positive
    height: Positive


class SquareObject(ObjectBase):
    kind: Literal["Square"]
    side_length: Positive


class PolygonObject(ObjectBase):
    kind: Literal["Polygon"]
    vertices: tuple[Vector, ...] = Field(min_length=3, max_length=100)


class LineObject(ObjectBase):
    kind: Literal["Line", "Arrow"]
    start: Vector
    end: Vector


class DotObject(ObjectBase):
    kind: Literal["Dot"]
    radius: Positive


class AxesObject(ObjectBase):
    kind: Literal["Axes"]
    x_range: tuple[FiniteFloat, FiniteFloat, Positive]
    y_range: tuple[FiniteFloat, FiniteFloat, Positive]
    x_length: Positive
    y_length: Positive


class GraphObject(ObjectBase):
    kind: Literal["Graph"]
    axes_element_id: str
    expression: str = Field(min_length=1, max_length=500)
    x_range: tuple[FiniteFloat, FiniteFloat]


ProgramObjectV2 = Annotated[
    TextObject
    | CircleObject
    | RectangleObject
    | SquareObject
    | PolygonObject
    | LineObject
    | DotObject
    | AxesObject
    | GraphObject,
    Field(discriminator="kind"),
]


class ProgramEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    operation: Literal[
        "add",
        "create",
        "write",
        "fade_in",
        "fade_out",
        "transform",
        "replacement_transform",
        "move_to",
        "shift",
        "scale",
        "rotate",
        "wait",
    ]
    start_seconds: FiniteFloat = Field(ge=0, le=600)
    end_seconds: FiniteFloat = Field(gt=0, le=600)
    source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    target_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    vector: Vector | None = None
    factor: Positive | None = None
    angle: FiniteFloat | None = None

    @model_validator(mode="after")
    def interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Event end must follow start")
        return self


class SceneProgram(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    scene_id: int = Field(ge=1)
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,199}$")
    duration_seconds: FiniteFloat = Field(gt=0, le=600)
    objects: tuple[ProgramObjectV2, ...] = Field(default_factory=tuple, max_length=200)
    events: tuple[ProgramEvent, ...] = Field(default_factory=tuple, max_length=300)
