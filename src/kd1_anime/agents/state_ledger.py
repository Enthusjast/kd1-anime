"""全片场景边界的语义状态账本。

ElementManifest 保存可重建的源代码快照；StateLedger 补充教学状态和边界
证据。它不执行生成代码，只保存经过结构化校验的事实，供后续场景和恢复
流程使用。
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LedgerElement(BaseModel):
    """场景边界上一个元素的语义和代码身份。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
    semantic_state: str = Field(default="", max_length=2_000)
    mathematical_state: str = Field(default="", max_length=3_000)
    color_key: str = Field(default="", max_length=100)
    anchor: str = Field(default="", max_length=500)
    active: bool = True
    required_next: bool = False
    source_scene_id: int = Field(ge=1)
    source_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")


class SceneBoundaryState(BaseModel):
    """一个 Scene 的开场/收场状态和渲染证据。"""

    model_config = ConfigDict(extra="forbid")

    scene_id: int = Field(ge=1)
    opening_element_ids: list[str] = Field(default_factory=list, max_length=100)
    closing_element_ids: list[str] = Field(default_factory=list, max_length=100)
    opening_math_state: str = Field(default="", max_length=4_000)
    closing_math_state: str = Field(default="", max_length=4_000)
    exported_code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    artifact_video_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    opening_frame_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    ending_frame_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")


class StateLedger(BaseModel):
    """可恢复的全片状态账本。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    current_scene_id: int | None = Field(default=None, ge=1)
    elements: list[LedgerElement] = Field(default_factory=list, max_length=200)
    boundaries: dict[int, SceneBoundaryState] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def validate_references(self) -> StateLedger:
        element_ids = [item.element_id for item in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("StateLedger.elements 的 element_id 必须唯一")
        known = set(element_ids)
        for scene_id, boundary in self.boundaries.items():
            if boundary.scene_id != scene_id:
                raise ValueError(f"StateLedger 边界 key {scene_id} 与 scene_id 不一致")
            for field_name in ("opening_element_ids", "closing_element_ids"):
                values = getattr(boundary, field_name)
                if len(values) != len(set(values)):
                    raise ValueError(f"StateLedger Scene {scene_id} 的 {field_name} 必须唯一")
            referenced = set(boundary.opening_element_ids) | set(boundary.closing_element_ids)
            missing = referenced - known
            if missing:
                raise ValueError(
                    f"StateLedger Scene {scene_id} 引用了不存在的元素: "
                    + ", ".join(sorted(missing))
                )
        if self.current_scene_id is not None and self.current_scene_id not in self.boundaries:
            raise ValueError("StateLedger.current_scene_id 没有对应的场景边界")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def for_elements(self, element_ids: set[str]) -> list[LedgerElement]:
        return [item for item in self.elements if item.element_id in element_ids]

    def update_scene(
        self,
        *,
        scene_id: int,
        elements: list[LedgerElement],
        opening_element_ids: list[str],
        closing_element_ids: list[str],
        opening_math_state: str = "",
        closing_math_state: str = "",
        exported_code_sha256: str = "",
        artifact_video_sha256: str = "",
    ) -> StateLedger:
        by_id = {item.element_id: item for item in self.elements}
        for item in elements:
            by_id[item.element_id] = item
        boundary = SceneBoundaryState(
            scene_id=scene_id,
            opening_element_ids=list(dict.fromkeys(opening_element_ids)),
            closing_element_ids=list(dict.fromkeys(closing_element_ids)),
            opening_math_state=opening_math_state,
            closing_math_state=closing_math_state,
            exported_code_sha256=exported_code_sha256,
            artifact_video_sha256=artifact_video_sha256,
        )
        return StateLedger.model_validate(
            {
                **self.model_dump(mode="python"),
                "current_scene_id": scene_id,
                "elements": list(by_id.values()),
                "boundaries": {**self.boundaries, scene_id: boundary},
            }
        )
