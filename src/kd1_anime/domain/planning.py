"""Planning contracts and deterministic normalization; no model clients."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from kd1_anime.domain.dependencies import SceneDependency
from kd1_anime.domain.program import SceneProgram
from kd1_anime.models import ValidatedModel as BaseModel


class GlobalVisualState(BaseModel):
    """全片共享、可被 Coder 直接执行的视觉规范。"""

    model_config = ConfigDict(extra="forbid")

    background: str = Field(default="#1C1C1C", min_length=1, max_length=500)
    colors: dict[str, str] = Field(
        default_factory=lambda: {
            "primary": "#58C4DD",
            "secondary": "#83C167",
            "highlight": "#FFFF00",
            "warning": "#FF6666",
            "foreground": "#FFFFFF",
        },
        max_length=50,
    )
    fonts: dict[str, str] = Field(
        default_factory=lambda: {
            "text": "Noto Sans CJK SC",
            "math": "STIX Two Math",
            "title": "Noto Sans CJK SC",
        },
        max_length=20,
    )
    font_sizes: dict[str, float] = Field(
        default_factory=lambda: {"title": 0.7, "body": 0.4, "formula": 0.8},
        max_length=20,
    )
    stroke_widths: dict[str, float] = Field(
        default_factory=lambda: {"default": 4.0, "highlight": 6.0},
        max_length=20,
    )
    layout_anchors: dict[str, str] = Field(
        default_factory=lambda: {
            "title": "top",
            "formula": "center",
            "main_content": "center",
        },
        max_length=30,
    )
    camera_language: str = Field(
        default="默认固定中景；只在关键揭示时推近或平移",
        min_length=1,
        max_length=4_000,
    )

    @field_validator("colors")
    @classmethod
    def validate_colors(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("全局调色板不能为空")
        for key, color in value.items():
            if not re.fullmatch(r"#[0-9A-Fa-f]{6,8}|[A-Za-z_][A-Za-z0-9_ -]{0,49}", color.strip()):
                raise ValueError(f"颜色 {key} 不是合法的十六进制或颜色名: {color!r}")
        return {str(key): color.strip() for key, color in value.items()}

    @field_validator("fonts")
    @classmethod
    def validate_fonts(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(not str(font).strip() for font in value.values()):
            raise ValueError("全局字体配置不能为空")
        return {str(key): str(font).strip() for key, font in value.items()}

    @field_validator("font_sizes", "stroke_widths")
    @classmethod
    def validate_positive_visual_values(cls, value: dict[str, float]) -> dict[str, float]:
        if any(float(number) <= 0 for number in value.values()):
            raise ValueError("字号和线宽必须为正数")
        return value


class VisualElementState(BaseModel):
    """一个可跨场景交接的语义视觉元素。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    role: str = Field(default="", max_length=500)
    kind: str = Field(default="Mobject", max_length=200)
    semantic_state: str = Field(default="", max_length=2_000)
    color_key: str = Field(default="", max_length=100)
    anchor: str = Field(default="", max_length=500)
    variable_name: str = Field(default="", pattern=r"^(?:[A-Za-z_][A-Za-z0-9_]*)?$")
    required: bool = True
    # elements_to_remove 需要说明退出原因；对 inherited/new 保持可选，兼容
    # 旧清单和模型只输出公共字段的情况。
    reason: str = Field(default="", max_length=2_000)


class ExtractedElement(BaseModel):
    """从已生成代码中提取的纯 Mobject 定义，持久化用于下一个场景。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
    code: str = Field(min_length=1, max_length=20_000)
    source_scene_id: int | None = Field(default=None, ge=1)
    source_code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")


class TimelineEvent(BaseModel):
    """可被确定性检查的时间线事件。"""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    start_seconds: float = Field(ge=0, le=600)
    end_seconds: float = Field(gt=0, le=600)
    action: str = Field(min_length=1, max_length=2_000)
    element_ids: list[str] = Field(default_factory=list, max_length=100)
    math_claim_ids: list[str] = Field(default_factory=list, max_length=100)
    pause_seconds: float = Field(default=0, ge=0, le=30)

    @model_validator(mode="before")
    @classmethod
    def normalize_event_id(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "event_id" not in data:
            data["event_id"] = data.get("id") or data.get("name") or "event_1"
        if "start_seconds" not in data and "start" in data:
            data["start_seconds"] = data["start"]
        if "start_seconds" not in data and "time" in data:
            data["start_seconds"] = data["time"]
        if "end_seconds" not in data and "end" in data:
            data["end_seconds"] = data["end"]
        if "end_seconds" not in data and "duration" in data and "start_seconds" in data:
            data["end_seconds"] = float(data["start_seconds"]) + float(data["duration"])
        if "action" not in data:
            data["action"] = data.get("event") or data.get("description") or "视觉事件"
        for alias in ("id", "name", "start", "end", "time", "duration", "event", "description"):
            data.pop(alias, None)
        return data

    @model_validator(mode="after")
    def validate_interval(self) -> TimelineEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("时间线事件的 end_seconds 必须大于 start_seconds")
        return self


class MathClaim(BaseModel):
    """计划中一个可以被复核的数学断言。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    statement: str = Field(min_length=1, max_length=3_000)
    expression_before: str = Field(default="", max_length=1_000)
    expression_after: str = Field(default="", max_length=1_000)
    relation: Literal["equivalent", "equals", "area", "definition", "inequality", "other"] = "other"
    justification: str = Field(default="", max_length=3_000)
    # 等价关系依赖定义域和前提。旧计划没有这些字段时保持空值，
    # 新计划必须在涉及除法、根号、对数或不等式时显式填写。
    domain: str = Field(default="", max_length=2_000)
    assumptions: list[str] = Field(default_factory=list, max_length=30)
    prerequisite_claim_ids: list[str] = Field(default_factory=list, max_length=50)
    teaching_note: str = Field(default="", max_length=2_000)

    @model_validator(mode="before")
    @classmethod
    def normalize_claim_id(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "claim_id" not in data:
            data["claim_id"] = data.get("id") or data.get("name") or "claim_1"
        if "statement" not in data:
            data["statement"] = data.get("claim") or data.get("description") or ""
        if "expression_before" not in data and "before" in data:
            data["expression_before"] = data["before"]
        if "expression_after" not in data and "after" in data:
            data["expression_after"] = data["after"]
        for alias in ("id", "name", "claim", "description", "before", "after"):
            data.pop(alias, None)
        return data


class GeometrySpec(BaseModel):
    """供确定性校验使用的几何规格；没有精确数据时不要声称面积守恒。"""

    model_config = ConfigDict(extra="forbid")

    geometry_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    shape: Literal[
        "polygon",
        "rectangle",
        "square",
        "triangle",
        "circle",
        "line",
        "point",
        "surface",
        "plane",
        "region",
        "other",
    ] = "other"
    vertices: list[list[float]] = Field(default_factory=list, max_length=100)
    declared_area: float | None = Field(default=None, ge=0)
    target_area: float | None = Field(default=None, ge=0)
    rotation_degrees: float = Field(default=0, ge=-3600, le=3600)
    coordinate_system: str = Field(default="", max_length=500)
    target_description: str = Field(default="", max_length=1_000)

    @model_validator(mode="before")
    @classmethod
    def normalize_geometry_id(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "geometry_id" not in data:
            data["geometry_id"] = (
                data.get("element_id") or data.get("id") or data.get("name") or "geometry_1"
            )
        data["geometry_id"] = str(data["geometry_id"])

        shape_aliases = {
            "poly": "polygon",
            "polygon": "polygon",
            "rectangle": "rectangle",
            "rect": "rectangle",
            "square": "square",
            "triangle": "triangle",
            "circle": "circle",
            "line": "line",
            "segment": "line",
            "point": "point",
            "surface": "surface",
            "plane": "plane",
            "region": "region",
            "area": "region",
        }
        raw_shape = data.get("shape", data.get("type", data.get("geometry_type")))
        if raw_shape is not None:
            normalized_shape = shape_aliases.get(str(raw_shape).strip().lower(), "other")
            data["shape"] = normalized_shape

        if "vertices" not in data or data.get("vertices") is None:
            coordinates = data.get("points", data.get("coordinates"))
            if isinstance(coordinates, (list, tuple)):
                if coordinates and all(
                    isinstance(point, (int, float)) and not isinstance(point, bool)
                    for point in coordinates
                ):
                    data["vertices"] = [list(coordinates)]
                else:
                    data["vertices"] = coordinates
        elif (
            isinstance(data["vertices"], (list, tuple))
            and data["vertices"]
            and all(
                isinstance(point, (int, float)) and not isinstance(point, bool)
                for point in data["vertices"]
            )
        ):
            data["vertices"] = [list(data["vertices"])]

        area_note = ""
        raw_area = data.get("declared_area")
        if raw_area is None and "area" in data:
            raw_area = data["area"]
        if isinstance(raw_area, (int, float)) and not isinstance(raw_area, bool):
            data["declared_area"] = raw_area
        elif raw_area not in (None, ""):
            area_note = f"原始面积描述：{raw_area}"
            data["declared_area"] = None
        if (
            "target_area" not in data
            and "target" in data
            and isinstance(data["target"], (int, float))
            and not isinstance(data["target"], bool)
        ):
            data["target_area"] = data["target"]
        elif "target_area" in data and not isinstance(data["target_area"], (int, float)):
            data["target_area"] = None

        raw_rotation = data.get("rotation_degrees", data.get("rotation"))
        if isinstance(raw_rotation, (int, float)) and not isinstance(raw_rotation, bool):
            data["rotation_degrees"] = raw_rotation
        elif isinstance(raw_rotation, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_rotation)
            data["rotation_degrees"] = float(match.group(0)) if match else 0.0

        if "coordinate_system" not in data:
            dimension = data.get("dimension")
            vertices = data.get("vertices") or []
            if isinstance(dimension, str) and dimension.strip():
                data["coordinate_system"] = dimension.strip()
            elif any(isinstance(point, (list, tuple)) and len(point) > 2 for point in vertices):
                data["coordinate_system"] = "3d"

        description_parts = [
            str(data.get("target_description", "")).strip(),
            str(data.get("description", "")).strip(),
            str(data.get("target_region", "")).strip(),
            area_note,
        ]
        description = "；".join(part for part in description_parts if part)
        if description:
            data["target_description"] = description[:1_000]

        vertices = data.get("vertices") or []
        is_3d = any(isinstance(point, (list, tuple)) and len(point) > 2 for point in vertices)
        if is_3d:
            # 现有鞋带公式只适用于二维多边形，不能把三维曲面/误差
            # 区域的数值面积伪装成已验证结果；保留描述供 Reviewer
            # 审查，但交给确定性编译器的面积字段必须为空。
            declared_area = data.get("declared_area")
            target_area = data.get("target_area")
            if declared_area is not None or target_area is not None:
                note = "三维几何面积未由二维鞋带公式自动核验"
                existing = str(data.get("target_description", "")).strip()
                data["target_description"] = "；".join(part for part in (existing, note) if part)[
                    :1_000
                ]
            data["declared_area"] = None
            data["target_area"] = None

        for alias in (
            "id",
            "name",
            "element_id",
            "points",
            "coordinates",
            "area",
            "target",
            "type",
            "geometry_type",
            "kind",
            "description",
            "target_region",
            "rotation",
            "dimension",
        ):
            data.pop(alias, None)
        return data


class LearningObjective(BaseModel):
    """一个可被画面证据验证的学习目标。"""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    outcome: str = Field(min_length=1, max_length=2_000)
    evidence: str = Field(default="", max_length=2_000)
    priority: Literal["core", "supporting"] = "core"


class MathEntity(BaseModel):
    """全片统一的数学符号/实体。"""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    notation: str = Field(min_length=1, max_length=500)
    meaning: str = Field(min_length=1, max_length=2_000)
    visual_role: str = Field(default="", max_length=1_000)
    color_key: str = Field(default="", max_length=100)


class TeachingEdge(BaseModel):
    """教学图谱中的有向依赖边。"""

    model_config = ConfigDict(extra="forbid")

    prerequisite_claim_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    dependent_claim_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    reason: str = Field(default="", max_length=1_000)


class LessonSpec(BaseModel):
    """全片唯一的教学和数学事实合同。"""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(default="", max_length=2_000)
    audience: str = Field(default="", max_length=2_000)
    prerequisites: list[str] = Field(default_factory=list, max_length=50)
    learning_objectives: list[LearningObjective] = Field(default_factory=list, max_length=30)
    entities: list[MathEntity] = Field(default_factory=list, max_length=100)
    claims: list[MathClaim] = Field(default_factory=list, max_length=200)
    misconceptions: list[str] = Field(default_factory=list, max_length=50)
    notation_rules: list[str] = Field(default_factory=list, max_length=50)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=50)
    requested_duration_min_seconds: float | None = Field(default=None, ge=0, le=3_600)
    requested_duration_max_seconds: float | None = Field(default=None, gt=0, le=3_600)
    scene_policy: Literal["minimum_visual_units", "user_defined", "single_visual_unit"] = (
        "minimum_visual_units"
    )

    @model_validator(mode="after")
    def validate_duration_range(self) -> LessonSpec:
        if (
            self.requested_duration_min_seconds is not None
            and self.requested_duration_max_seconds is not None
            and self.requested_duration_min_seconds > self.requested_duration_max_seconds
        ):
            raise ValueError("LessonSpec 的最小时长不能大于最大时长")
        return self


def repair_obvious_math_contradictions(
    lesson_spec: LessonSpec,
    user_prompt: str,
) -> tuple[LessonSpec, list[str]]:
    """修正可由给定矩阵直接判定的教学合同矛盾。

    用户描述应当被尊重，但不能让一个与明确矩阵运算冲突的自然语言
    结论进入全片唯一的数学事实合同。这里只处理一个证据充分、语义
    边界清晰的基础情形：剪切矩阵 ``[[1, 1], [0, 1]]`` 的 x 轴是不变
    方向。其它无法由轻量规则可靠判断的数学内容仍交给 Planner 和
    Plan Reviewer，不在这里猜测或重写。
    """

    prompt_text = str(user_prompt or "")
    has_unit_shear = bool(
        re.search(
            r"剪切矩阵[\s\S]{0,240}(?:1\s*&\s*1|1\s*,\s*1)[\s\S]{0,120}"
            r"(?:0\s*&\s*1|0\s*,\s*1)",
            prompt_text,
            flags=re.IGNORECASE,
        )
    )
    if not has_unit_shear:
        return lesson_spec, []

    false_phrases = (
        "没有任何向量保持在原直线上",
        "无实特征向量",
        "无方向不变的向量",
    )
    replacement = "除 x 轴方向外，没有其它向量保持在原直线上"
    corrected_claims: list[MathClaim] = []
    repairs: list[str] = []
    for claim in lesson_spec.claims:
        claim_data = claim.model_dump(mode="python")
        changed = False
        for field_name in ("statement", "justification", "teaching_note"):
            value = str(claim_data.get(field_name) or "")
            if not any(phrase in value for phrase in false_phrases):
                continue
            updated = value
            updated = updated.replace(
                "没有任何向量保持在原直线上（无实特征向量）",
                "只有 x 轴方向的向量保持在原直线上（实特征向量方向为 (1,0)）",
            )
            updated = updated.replace("没有任何向量保持在原直线上", replacement)
            updated = updated.replace("无实特征向量", "存在实特征向量 (1,0)，但没有其它不变方向")
            updated = updated.replace("无方向不变的向量", "除 x 轴方向外没有其它方向保持不变")
            if updated != value:
                claim_data[field_name] = updated
                changed = True
        corrected_claims.append(MathClaim.model_validate(claim_data))
        if changed:
            repairs.append(f"已校正 {claim.claim_id}：剪切矩阵的 x 轴不变方向")

    def repair_text(value: str) -> str:
        updated = str(value or "")
        updated = updated.replace(
            "没有任何向量保持在原直线上（无实特征向量）", "只有 x 轴方向保持在原直线上"
        )
        updated = updated.replace("没有任何向量保持在原直线上", replacement)
        updated = updated.replace("无实特征向量", "存在实特征向量 (1,0)，但没有其它不变方向")
        updated = updated.replace("无方向不变的向量", "除 x 轴方向外没有其它方向保持不变")
        return updated

    objectives = [
        objective.model_copy(
            update={
                "outcome": repair_text(objective.outcome),
                "evidence": repair_text(objective.evidence),
            }
        )
        for objective in lesson_spec.learning_objectives
    ]
    misconceptions = [repair_text(value) for value in lesson_spec.misconceptions]
    acceptance_criteria = [repair_text(value) for value in lesson_spec.acceptance_criteria]
    if not repairs:
        return lesson_spec, []
    return (
        lesson_spec.model_copy(
            update={
                "claims": corrected_claims,
                "learning_objectives": objectives,
                "misconceptions": misconceptions,
                "acceptance_criteria": acceptance_criteria,
            }
        ),
        repairs,
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> LessonSpec:
        for field_name in ("learning_objectives", "entities", "claims"):
            values = getattr(self, field_name)
            id_field = {
                "learning_objectives": "objective_id",
                "entities": "entity_id",
                "claims": "claim_id",
            }[field_name]
            ids = [getattr(item, id_field) for item in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"LessonSpec.{field_name} 的 {id_field} 必须唯一")
        return self


class TeachingGraph(BaseModel):
    """连接数学断言、教学顺序和场景分配的全片图谱。"""

    model_config = ConfigDict(extra="forbid")

    claim_order: list[str] = Field(default_factory=list, max_length=200)
    edges: list[TeachingEdge] = Field(default_factory=list, max_length=300)
    scene_claims: dict[int, list[str]] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def validate_unique_assignments(self) -> TeachingGraph:
        if len(self.claim_order) != len(set(self.claim_order)):
            raise ValueError("TeachingGraph.claim_order 的 claim_id 必须唯一")
        for scene_id, claim_ids in self.scene_claims.items():
            if len(claim_ids) != len(set(claim_ids)):
                raise ValueError(f"TeachingGraph.scene_claims[{scene_id}] 的 claim_id 必须唯一")
        return self


class PlanningDraft(BaseModel):
    """概要阶段的完整结构化输出；items 保持与旧 Outline API 一致。"""

    model_config = ConfigDict(extra="forbid")

    lesson_spec: LessonSpec = Field(default_factory=LessonSpec)
    teaching_graph: TeachingGraph = Field(default_factory=TeachingGraph)
    items: list[SceneOutline] = Field(min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_outline_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("items"):
            data["items"] = data.get("outlines") or data.get("scene_outlines")
        data.pop("outlines", None)
        data.pop("scene_outlines", None)
        return data


def compact_lesson_spec(
    lesson_spec: LessonSpec | None,
    *,
    claim_ids: set[str] | None = None,
    max_chars: int = 20_000,
) -> str:
    """为下游 Agent 构造有界的教学合同摘要。"""

    if lesson_spec is None:
        return "{}"
    data = lesson_spec.model_dump(mode="json")
    claims = data.get("claims", [])
    if claim_ids:
        selected = [item for item in claims if item.get("claim_id") in claim_ids]
        selected_ids = {item.get("claim_id") for item in selected}
        # 递归带上前置断言，避免只注入一层导致 Coder/Reviewer 看不到
        # 更早的定义域、定义或等价变换条件。
        by_id = {item.get("claim_id"): item for item in claims}
        pending = list(selected)
        while pending:
            current = pending.pop()
            for prerequisite in current.get("prerequisite_claim_ids", []):
                if prerequisite in selected_ids or prerequisite not in by_id:
                    continue
                prerequisite_item = by_id[prerequisite]
                selected.append(prerequisite_item)
                selected_ids.add(prerequisite)
                pending.append(prerequisite_item)
        claims = selected
    data["claims"] = claims[:40]
    data["entities"] = data.get("entities", [])[:40]
    data["learning_objectives"] = data.get("learning_objectives", [])[:20]
    for claim in data["claims"]:
        for key in (
            "statement",
            "expression_before",
            "expression_after",
            "justification",
            "domain",
            "teaching_note",
        ):
            if isinstance(claim.get(key), str):
                claim[key] = claim[key][:1_500]
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        payload if len(payload) <= max_chars else payload[:max_chars] + "\n...[教学合同摘要已裁剪]"
    )


def compact_teaching_graph(
    teaching_graph: TeachingGraph | None,
    *,
    scene_id: int | None = None,
    max_chars: int = 10_000,
) -> str:
    """构造不会挤占代码/合同预算的教学图谱摘要。"""

    if teaching_graph is None:
        return "{}"
    data = teaching_graph.model_dump(mode="json")
    if scene_id is not None:
        data["scene_claims"] = {str(scene_id): data.get("scene_claims", {}).get(scene_id, [])}
    data["edges"] = data.get("edges", [])[:80]
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        payload if len(payload) <= max_chars else payload[:max_chars] + "\n...[教学图谱摘要已裁剪]"
    )


class SceneHandoff(BaseModel):
    """场景边界上的元素生命周期合同。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(default="", pattern=r"^(?:[A-Za-z_][A-Za-z0-9_]*)?$")
    action: Literal["inherit", "keep", "create", "remove"] = "keep"
    semantic_state: str = Field(default="", max_length=2_000)
    transition: str = Field(default="", max_length=2_000)

    @model_validator(mode="before")
    @classmethod
    def normalize_handoff(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "element_id" not in data:
            data["element_id"] = data.get("id") or data.get("name") or "element_1"
        if data.get("action") == "persistent":
            data["action"] = "keep"
        if "semantic_state" not in data and "state" in data:
            data["semantic_state"] = data["state"]
        if "transition" not in data and "transition_out" in data:
            data["transition"] = data["transition_out"]
        for alias in ("id", "name", "state", "transition_out"):
            data.pop(alias, None)
        return data


class ElementManifestEntry(BaseModel):
    """运行级连续性清单中的一个最终元素。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
    kind: str = Field(default="Mobject", max_length=200)
    dependencies: list[str] = Field(default_factory=list, max_length=100)
    semantic_state: str = Field(default="", max_length=2_000)
    source_scene_id: int = Field(ge=1)
    source_code: str = Field(min_length=1, max_length=20_000)
    source_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ElementManifest(BaseModel):
    """供相邻场景消费的最小、可追踪元素状态快照。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    entries: list[ElementManifestEntry] = Field(default_factory=list, max_length=200)
    scene_exports: dict[int, list[str]] = Field(default_factory=dict, max_length=100)
    last_scene_id: int | None = Field(default=None, ge=1)

    def digest(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def for_elements(self, element_ids: set[str]) -> list[ElementManifestEntry]:
        return [entry for entry in self.entries if entry.element_id in element_ids]

    def update_scene(
        self,
        plan: ScenePlan,
        elements: list[ExtractedElement],
    ) -> ElementManifest:
        """以本场景最终导出区替换对应快照，移除已明确退出的元素。"""

        removed_ids = {item.element_id for item in plan.elements_to_remove}
        declarations = {
            item.element_id: item
            for item in [*plan.inherited_elements, *plan.new_elements]
            if item.element_id not in removed_ids
        }
        entry_by_id = {entry.element_id: entry for entry in self.entries}
        exported_ids: list[str] = []
        for element in elements:
            declaration = declarations.get(element.element_id)
            try:
                tree = ast.parse(element.code)
                bound = {
                    node.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                }
                dependencies = sorted(
                    {
                        node.id
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in bound
                        and node.id not in {"self", "config", "tex_template"}
                    }
                )
            except SyntaxError:
                dependencies = []
            entry_by_id[element.element_id] = ElementManifestEntry(
                element_id=element.element_id,
                variable_name=element.variable_name,
                kind=declaration.kind if declaration is not None else "Mobject",
                dependencies=dependencies,
                semantic_state=(declaration.semantic_state if declaration is not None else ""),
                source_scene_id=plan.scene_id,
                source_code=element.code,
                source_code_sha256=hashlib.sha256(element.code.encode("utf-8")).hexdigest(),
            )
            exported_ids.append(element.element_id)
        for element_id in removed_ids:
            entry_by_id.pop(element_id, None)
        scene_exports = dict(self.scene_exports)
        scene_exports[plan.scene_id] = exported_ids
        return self.model_copy(
            update={
                "entries": list(entry_by_id.values()),
                "scene_exports": scene_exports,
                "last_scene_id": plan.scene_id,
            }
        )


def _normalize_element_list(value):
    """兼容模型把元素写成字符串、对象或对象数组的常见输出。"""

    if value is None:
        return []
    values = [value] if isinstance(value, (str, dict)) else value
    if not isinstance(values, list):
        return values
    normalized = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, str):
            normalized.append(
                {
                    "element_id": f"element_{index}",
                    "semantic_state": item,
                }
            )
        elif isinstance(item, dict):
            data = dict(item)
            data.setdefault("element_id", data.get("id") or data.get("name") or f"element_{index}")
            if "semantic_state" not in data:
                data["semantic_state"] = data.get("state", data.get("description", ""))
            normalized.append(data)
        elif isinstance(item, BaseModel):
            normalized.append(item.model_dump(mode="python"))
        else:
            normalized.append({"element_id": f"element_{index}", "semantic_state": str(item)})
    return normalized


def _normalize_structured_list(value, kind: str):
    """把模型偶尔输出的简写字符串转成可审查的结构化对象。"""

    if value is None:
        return []
    values = [value] if isinstance(value, (str, dict)) else value
    if not isinstance(values, list):
        return values
    result = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, str):
            if kind == "math_claim":
                result.append({"claim_id": f"claim_{index}", "statement": item})
            elif kind == "geometry":
                result.append({"geometry_id": f"geometry_{index}", "target_description": item})
            else:
                result.append(
                    {
                        "element_id": f"element_{index}",
                        "semantic_state": item,
                    }
                )
        else:
            result.append(item)
    return result


class ScenePlan(BaseModel):
    """单个场景的完整导演规划。"""

    model_config = ConfigDict(extra="forbid")

    dependencies: list[SceneDependency] = Field(default_factory=list, max_length=100)
    dependencies_complete: bool = False
    program: SceneProgram | None = None
    scene_id: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    duration_seconds: float = Field(gt=0, le=600)
    purpose: str = Field(min_length=1, max_length=5_000)
    math_concept: str = Field(min_length=1, max_length=5_000)
    # 概要阶段确定的教学归属；Detail 只能补充视觉实现，不能新增未声明的
    # 数学断言或改变 Scene 的视觉单元身份。
    claim_ids: list[str] = Field(default_factory=list, max_length=100)
    visual_unit_id: str = Field(default="", max_length=200)
    teaching_role: str = Field(default="", max_length=1_000)
    visual_design: str = Field(min_length=1, max_length=20_000)
    camera_movement: str = Field(min_length=1, max_length=10_000)
    visual_flow: list[str] = Field(min_length=1, max_length=100)
    key_moments: list[str] = Field(min_length=1, max_length=100)
    computation: str = Field(min_length=1, max_length=20_000)
    # 跨场景连续性合同。旧清单/旧测试没有这些字段时使用空列表，恢复仍然兼容；
    # 新运行会由 Detail Prompt 填充，并在连续性审查阶段校验。
    persistent_elements: list[str] = Field(default_factory=list, max_length=100)
    opening_state: list[str] = Field(default_factory=list, max_length=100)
    closing_state: list[str] = Field(default_factory=list, max_length=100)
    transition_in: str = Field(default="", max_length=10_000)
    transition_out: str = Field(default="", max_length=10_000)
    continuity_references: list[str] = Field(default_factory=list, max_length=100)
    global_visual_state: GlobalVisualState = Field(default_factory=GlobalVisualState)
    inherited_elements: list[VisualElementState] = Field(default_factory=list, max_length=100)
    elements_to_remove: list[VisualElementState] = Field(default_factory=list, max_length=100)
    new_elements: list[VisualElementState] = Field(default_factory=list, max_length=100)
    timeline: list[TimelineEvent] = Field(default_factory=list, max_length=100)
    math_claims: list[MathClaim] = Field(default_factory=list, max_length=100)
    geometry_specs: list[GeometrySpec] = Field(default_factory=list, max_length=100)
    handoff: list[SceneHandoff] = Field(default_factory=list, max_length=100)

    @field_validator("inherited_elements", "elements_to_remove", "new_elements", mode="before")
    @classmethod
    def normalize_elements(cls, value):
        return _normalize_element_list(value)

    @field_validator("claim_ids")
    @classmethod
    def validate_claim_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("ScenePlan.claim_ids 必须唯一")
        return value

    @field_validator("math_claims", "geometry_specs", "handoff", mode="before")
    @classmethod
    def normalize_structured_fields(cls, value, info):
        kind = {
            "math_claims": "math_claim",
            "geometry_specs": "geometry",
            "handoff": "handoff",
        }[info.field_name]
        return _normalize_structured_list(value, kind)


class SceneOutline(BaseModel):
    """阶段 1 输出：场景概要。"""

    model_config = ConfigDict(extra="forbid")

    scene_id: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    duration_seconds: float = Field(gt=0, le=600)
    purpose: str = Field(min_length=1, max_length=5_000)
    math_concept: str = Field(min_length=1, max_length=5_000)
    claim_ids: list[str] = Field(default_factory=list, max_length=100)
    visual_unit_id: str = Field(default="", max_length=200)
    teaching_role: str = Field(default="", max_length=1_000)

    @field_validator("claim_ids")
    @classmethod
    def validate_claim_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("SceneOutline.claim_ids 必须唯一")
        return value


class SceneDetail(BaseModel):
    """阶段 2 输出：单个场景的导演细节。"""

    model_config = ConfigDict(extra="forbid")

    dependencies: list[SceneDependency] = Field(default_factory=list, max_length=100)
    dependencies_complete: bool = False
    program: SceneProgram | None = None
    visual_design: str = Field(min_length=1, max_length=20_000)
    camera_movement: str = Field(min_length=1, max_length=10_000)
    visual_flow: list[str] = Field(min_length=1, max_length=100)
    key_moments: list[str] = Field(min_length=1, max_length=100)
    computation: str = Field(min_length=1, max_length=20_000)
    persistent_elements: list[str] = Field(default_factory=list, max_length=100)
    opening_state: list[str] = Field(default_factory=list, max_length=100)
    closing_state: list[str] = Field(default_factory=list, max_length=100)
    transition_in: str = Field(default="", max_length=10_000)
    transition_out: str = Field(default="", max_length=10_000)
    continuity_references: list[str] = Field(default_factory=list, max_length=100)
    global_visual_state: GlobalVisualState = Field(default_factory=GlobalVisualState)
    inherited_elements: list[VisualElementState] = Field(default_factory=list, max_length=100)
    elements_to_remove: list[VisualElementState] = Field(default_factory=list, max_length=100)
    new_elements: list[VisualElementState] = Field(default_factory=list, max_length=100)
    timeline: list[TimelineEvent] = Field(default_factory=list, max_length=100)
    math_claims: list[MathClaim] = Field(default_factory=list, max_length=100)
    geometry_specs: list[GeometrySpec] = Field(default_factory=list, max_length=100)
    handoff: list[SceneHandoff] = Field(default_factory=list, max_length=100)

    @field_validator("visual_design", "computation", mode="before")
    @classmethod
    def ensure_string(cls, v):
        """LLM 有时返回对象而非字符串，自动转换为 JSON 字符串"""
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False, indent=2)
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False, indent=2)
        return v

    @field_validator(
        "key_moments",
        "visual_flow",
        "persistent_elements",
        "opening_state",
        "closing_state",
        "continuity_references",
        mode="before",
    )
    @classmethod
    def ensure_string_list(cls, v):
        """LLM 有时返回对象数组而非字符串数组，自动转换为字符串列表"""
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return [json.dumps(v, ensure_ascii=False)]
        if isinstance(v, list):
            converted = []
            for item in v:
                if isinstance(item, str):
                    converted.append(item)
                elif isinstance(item, dict):
                    # 将 {time, event, pause} 合并为单个字符串
                    parts = []
                    for key in ("time", "event", "pause", "description"):
                        if item.get(key):
                            parts.append(str(item[key]))
                    if parts:
                        converted.append(" - ".join(parts))
                    else:
                        converted.append(json.dumps(item, ensure_ascii=False))
                else:
                    converted.append(str(item))
            return converted
        return v

    @field_validator("inherited_elements", "elements_to_remove", "new_elements", mode="before")
    @classmethod
    def normalize_elements(cls, value):
        return _normalize_element_list(value)

    @field_validator("math_claims", "geometry_specs", "handoff", mode="before")
    @classmethod
    def normalize_structured_fields(cls, value, info):
        kind = {
            "math_claims": "math_claim",
            "geometry_specs": "geometry",
            "handoff": "handoff",
        }[info.field_name]
        return _normalize_structured_list(value, kind)


class ContinuityBible(BaseModel):
    """整部动画共享的视觉、数学和叙事规范。"""

    model_config = ConfigDict(extra="forbid")

    # 结构化视觉配置供 Coder 直接使用；下面的 legacy 字段保留，便于恢复
    # schema 2 manifest 和兼容已有 LLM 输出。
    global_visual_state: GlobalVisualState = Field(default_factory=GlobalVisualState)

    @model_validator(mode="before")
    @classmethod
    def derive_visual_state_from_legacy_fields(cls, value):
        """让旧版仅包含 background/camera_language 的圣经也能提供结构化配置。"""

        if not isinstance(value, dict) or "global_visual_state" in value:
            return value
        visual_state = {
            "background": value.get("background", "#1C1C1C"),
            "camera_language": value.get(
                "camera_language", "默认固定中景；只在关键揭示时推近或平移"
            ),
        }
        palette = value.get("palette") or []
        color_keys = ("primary", "secondary", "highlight", "warning")
        colors = dict(GlobalVisualState().colors)
        for key, palette_item in zip(color_keys, palette, strict=False):
            match = re.search(r"#[0-9A-Fa-f]{6}", str(palette_item))
            if match:
                colors[key] = match.group(0)
        visual_state["colors"] = colors
        updated = dict(value)
        updated["global_visual_state"] = visual_state
        return updated

    background: str = Field(default="#1C1C1C 深灰背景", min_length=1, max_length=2_000)
    palette: list[str] = Field(
        default_factory=lambda: [
            "主色 #58C4DD（已知/输入）",
            "辅色 #83C167（结果/输出）",
            "强调色 #FFFF00（关键揭示）",
            "警告色 #FF6666（错误/对消）",
        ],
        max_length=50,
    )
    typography: str = Field(
        default="中文使用 Noto Sans CJK SC；标题、正文、公式使用固定字号层级，避免场景间跳变",
        min_length=1,
        max_length=4_000,
    )
    layout: str = Field(
        default="16:9 画布；标题区固定在顶部；主体对象保持在安全边距内；公式区与图形区使用稳定锚点",
        min_length=1,
        max_length=4_000,
    )
    math_notation: str = Field(
        default="变量命名、上下标、等号链和颜色语义全片统一；后续场景沿用前一场景已定义的符号",
        min_length=1,
        max_length=5_000,
    )
    persistent_elements: list[str] = Field(
        default_factory=lambda: ["顶部章节标题", "当前核心公式", "变量颜色语义"],
        max_length=100,
    )
    camera_language: str = Field(
        default="默认固定中景；只在关键揭示时推近或平移，镜头变化必须服务于焦点转移",
        min_length=1,
        max_length=4_000,
    )
    narrative_arc: str = Field(
        default="从问题建立到逐步推导，最后保留结论并完成总结，不在场景边界重复开场",
        min_length=1,
        max_length=4_000,
    )
    transition_rules: list[str] = Field(
        default_factory=lambda: [
            "下一场景开头先接管上一场景结束时保留的对象或公式",
            "优先使用对象变换和焦点移动，不无故清空画面后重新绘制",
            "每个场景结束都要明确交接给下一场景的数学状态",
        ],
        max_length=100,
    )


PlanningDraft.model_rebuild()
