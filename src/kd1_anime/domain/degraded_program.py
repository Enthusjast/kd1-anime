"""Explicitly degraded v1 fallback program contracts."""

from typing import Literal

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel

ProgramOperation = Literal[
    "add",
    "create",
    "write",
    "fade_in",
    "transform",
    "replacement_transform",
    "animate",
    "fade_out",
    "wait",
]


class ProgramObject(BaseModel):
    """SceneProgram 中的一个受约束对象。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
    kind: str = Field(default="Text", min_length=1, max_length=80)
    label: str = Field(default="", max_length=500)
    color_key: str = Field(default="primary", max_length=80)
    initially_active: bool = False
    exported: bool = False
    removed: bool = False
    dependencies: list[str] = Field(default_factory=list, max_length=50)


class ProgramAnimation(BaseModel):
    """SceneProgram 中的一条动画操作。"""

    model_config = ConfigDict(extra="forbid")

    operation: ProgramOperation
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    target_ids: list[str] = Field(default_factory=list, max_length=20)
    run_time: float = Field(default=1.0, gt=0, le=30)


class DegradedSceneProgram(BaseModel):
    """可编译为单个 Manim Scene 的结构化程序。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    degraded: Literal[True] = True
    scene_id: int = Field(ge=1)
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,199}$")
    parent: Literal["Scene", "ThreeDScene", "MovingCameraScene"] = "Scene"
    objects: list[ProgramObject] = Field(default_factory=list, max_length=200)
    animations: list[ProgramAnimation] = Field(default_factory=list, max_length=300)
    summary: str = Field(default="", max_length=500)
    wait_seconds: float = Field(default=1.0, ge=0, le=30)
