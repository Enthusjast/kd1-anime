"""Closed technical contracts shared by planning, validation and rendering."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from kd1_anime.models import ValidatedModel as BaseModel

LifecycleAction = Literal["define", "introduce", "update", "keep", "remove"]
SemanticAnimationAction = Literal["introduce", "update", "remove", "camera", "hold"]


class TechnicalObject(BaseModel):
    """一个场景内可被动画引用的对象。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    variable_name: str = Field(default="", pattern=r"^(?:[A-Za-z_][A-Za-z0-9_]*)?$")
    constructor: str = Field(default="Mobject", min_length=1, max_length=200)
    dependencies: list[str] = Field(default_factory=list, max_length=100)
    initial_state: str = Field(default="", max_length=2_000)
    final_state: str = Field(default="", max_length=2_000)
    visual_role: str = Field(default="", max_length=1_000)
    z_index: int = Field(default=0, ge=-10_000, le=10_000)
    estimated_width: float | None = Field(default=None, gt=0, le=100)
    estimated_height: float | None = Field(default=None, gt=0, le=100)
    lifecycle: list[LifecycleAction] = Field(default_factory=list, max_length=30)
    initially_active: bool = False
    exported: bool = False


class TechnicalHandoff(BaseModel):
    """供下一个场景使用的结构化技术边界。"""

    model_config = ConfigDict(extra="forbid")

    source_scene_id: int = Field(ge=1)
    elements: list[TechnicalObject] = Field(default_factory=list, max_length=100)


class TechnicalAnimation(BaseModel):
    """一条与具体 Manim 动画类解耦的状态事件。

    Coder 可以自由选择适合当前画面的 Animation 或 ``.animate`` 调用；
    这里仅描述对象状态如何变化。这样新增的 Manim 动画无需先加入
    技术计划的枚举表，也不会因为名称未被静态分析器认识而被错误拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
    start_seconds: float = Field(ge=0, le=600)
    end_seconds: float = Field(gt=0, le=600)
    semantic_action: SemanticAnimationAction
    source_element_ids: list[str] = Field(default_factory=list, max_length=50)
    target_element_ids: list[str] = Field(default_factory=list, max_length=50)
    create_element_ids: list[str] = Field(default_factory=list, max_length=50)
    remove_element_ids: list[str] = Field(default_factory=list, max_length=50)
    claim_ids: list[str] = Field(default_factory=list, max_length=50)
    api_notes: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def validate_interval(self) -> TechnicalAnimation:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("TechnicalAnimation 的 end_seconds 必须大于 start_seconds")
        return self


class TechnicalLayout(BaseModel):
    """技术布局约束。"""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="使用相对定位和稳定锚点", max_length=4_000)
    anchors: dict[str, str] = Field(default_factory=dict, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    safe_margin: float = Field(default=0.5, ge=0, le=20)
    minimum_spacing: float = Field(default=0.3, ge=0, le=20)


class TechnicalLatex(BaseModel):
    """Tex/MathTex 的实现约束。"""

    model_config = ConfigDict(extra="forbid")

    required: bool = False
    template_name: str = Field(default="tex_template", max_length=100)
    compiler: str = Field(default="xelatex", max_length=100)
    output_format: str = Field(default=".xdv", max_length=20)
    preamble_packages: list[str] = Field(default_factory=list, max_length=30)
    substrings_to_isolate: list[str] = Field(default_factory=list, max_length=100)
    expected_part_counts: dict[str, int] = Field(default_factory=dict, max_length=100)
    notes: str = Field(default="", max_length=4_000)


class TechnicalSpec(BaseModel):
    """Coder 必须遵守的单场景技术合同。"""

    model_config = ConfigDict(extra="forbid")

    # 该字段是直接替换旧技术合同的明确版本闸门。旧合同没有该字段，
    # 恢复时会被标记为 stale 而不是悄悄按新语义解释。
    contract_version: Literal[2] = 2
    scene_id: int = Field(ge=1)
    renderer: Literal["cairo", "opengl"] = "cairo"
    objects: list[TechnicalObject] = Field(default_factory=list, max_length=200)
    animations: list[TechnicalAnimation] = Field(default_factory=list, max_length=300)
    layout: TechnicalLayout = Field(default_factory=TechnicalLayout)
    latex: TechnicalLatex = Field(default_factory=TechnicalLatex)
    export_element_ids: list[str] = Field(default_factory=list, max_length=100)
    removed_element_ids: list[str] = Field(default_factory=list, max_length=100)
    implementation_notes: list[str] = Field(default_factory=list, max_length=100)
    # 跨场景连续性由结构化技术边界传递；实际生成代码只作为后置验证材料。
    handoff_in: TechnicalHandoff | None = None
    handoff_out: TechnicalHandoff | None = None
