"""受约束的场景中间表示和 Manim 源码编译器。

SceneProgram 是生成失败时的确定性后备路径：它只允许少量经过验证的
Manim 构造器和动画操作，避免让 LLM 同时承担 Python 语法与对象生命周期。
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.agents.planner import ScenePlan, VisualElementState
from kd1_anime.agents.technical_planner import TechnicalSpec

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


class SceneProgram(BaseModel):
    """可编译为单个 Manim Scene 的结构化程序。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    scene_id: int = Field(ge=1)
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,199}$")
    parent: Literal["Scene", "ThreeDScene", "MovingCameraScene"] = "Scene"
    objects: list[ProgramObject] = Field(default_factory=list, max_length=200)
    animations: list[ProgramAnimation] = Field(default_factory=list, max_length=300)
    summary: str = Field(default="", max_length=500)
    wait_seconds: float = Field(default=1.0, ge=0, le=30)


class SceneProgramCompileError(ValueError):
    """SceneProgram 无法安全编译。"""


_SUPPORTED_KINDS = {
    "Text",
    "MathTex",
    "Tex",
    "Circle",
    "Square",
    "Rectangle",
    "Triangle",
    "Dot",
    "Line",
    "Arrow",
    "Polygon",
    "VGroup",
    "Axes",
    "ThreeDAxes",
    "Surface",
}


def _safe_variable(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", value or "element")
    if not candidate or not (candidate[0].isalpha() or candidate[0] == "_"):
        candidate = "element_" + candidate
    return candidate


def _quote(value: str) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _color_expression(color_key: str) -> str:
    key = _safe_variable(color_key or "primary")
    if key not in {"primary", "secondary", "highlight"}:
        key = "primary"
    return f'COLORS["{key}"]'


def _object_expression(item: ProgramObject, variables: set[str]) -> str:
    kind = item.kind if item.kind in _SUPPORTED_KINDS else "Text"
    color = _color_expression(item.color_key)
    label = item.label[:180] or item.element_id
    if kind in {"MathTex", "Tex"}:
        # 从合同生成的后备对象优先使用 Text，避免把自然语言语义状态
        # 误当成 LaTeX；正常 LLM 代码仍可使用严格的 MathTex 模板。
        return f"Text({_quote(label)}, color={color})"
    if kind == "Text":
        return f"Text({_quote(label)}, color={color}, font_size=28)"
    if kind == "Circle":
        return f"Circle(radius=1.0, color={color})"
    if kind == "Square":
        return f"Square(side_length=2.0, color={color})"
    if kind == "Rectangle":
        return f"Rectangle(width=3.0, height=2.0, color={color})"
    if kind == "Triangle":
        return f"Triangle(color={color})"
    if kind == "Dot":
        return f"Dot(color={color})"
    if kind in {"Line", "Arrow"}:
        return f"{kind}(LEFT * 1.5, RIGHT * 1.5, color={color})"
    if kind == "Polygon":
        return f"Polygon(LEFT * 1.5, RIGHT * 1.5, UP * 1.5, color={color})"
    if kind == "VGroup":
        missing = [name for name in item.dependencies if name not in variables]
        if missing:
            raise SceneProgramCompileError(
                f"VGroup {item.variable_name} 依赖未定义对象: {', '.join(missing)}"
            )
        return f"VGroup({', '.join(item.dependencies)})"
    if kind == "Axes":
        return "Axes(x_range=[-4, 4, 1], y_range=[-3, 3, 1])"
    if kind == "ThreeDAxes":
        return "ThreeDAxes()"
    if kind == "Surface":
        return "Surface(lambda u, v: np.array([u, v, 0]), u_range=[-2, 2], v_range=[-2, 2], resolution=(10, 10))"
    raise SceneProgramCompileError(f"不支持的 SceneProgram 对象类型: {item.kind}")


def build_scene_program_from_contract(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
) -> SceneProgram:
    """从已批准的 ScenePlan/TechnicalSpec 构造最小确定性程序。"""

    technical_objects = {
        item.element_id: item for item in (technical_spec.objects if technical_spec else ())
    }
    removed_ids = {item.element_id for item in scene_plan.elements_to_remove}
    declared = [*scene_plan.inherited_elements, *scene_plan.new_elements]
    objects: list[ProgramObject] = []
    seen: set[str] = set()

    def append(item: VisualElementState, *, removed: bool = False) -> None:
        if item.element_id in seen:
            return
        seen.add(item.element_id)
        technical = technical_objects.get(item.element_id)
        variable = (
            (technical.variable_name if technical else "")
            or item.variable_name
            or _safe_variable(item.element_id)
        )
        kind = technical.constructor if technical else item.kind
        objects.append(
            ProgramObject(
                element_id=item.element_id,
                variable_name=variable,
                kind=kind,
                label=item.role or item.semantic_state or item.element_id,
                color_key=item.color_key or "primary",
                initially_active=bool(technical and technical.initially_active),
                exported=item.required and not removed,
                removed=removed,
            )
        )

    for item in scene_plan.elements_to_remove:
        if item.element_id in technical_objects or item.element_id in {
            element.element_id for element in scene_plan.inherited_elements
        }:
            append(item, removed=True)
    for item in declared:
        if item.element_id not in removed_ids:
            append(item)

    animations: list[ProgramAnimation] = []
    for item in objects:
        if item.removed:
            animations.extend(
                [
                    ProgramAnimation(operation="add", source_ids=[item.element_id], run_time=0.1),
                    ProgramAnimation(
                        operation="fade_out", source_ids=[item.element_id], run_time=0.3
                    ),
                ]
            )
        elif item.initially_active:
            animations.append(
                ProgramAnimation(operation="add", source_ids=[item.element_id], run_time=0.1)
            )
        elif item.exported:
            animations.append(
                ProgramAnimation(operation="fade_in", source_ids=[item.element_id], run_time=0.3)
            )
    return SceneProgram(
        scene_id=scene_plan.scene_id,
        class_name=f"Scene{scene_plan.scene_id}",
        parent="ThreeDScene"
        if any(item.kind in {"Surface", "ThreeDAxes"} for item in objects)
        else "Scene",
        objects=objects,
        animations=animations,
        summary=(scene_plan.math_concept + "：" + scene_plan.computation)[:500],
        wait_seconds=1.0,
    )


def compile_scene_program(program: SceneProgram, scene_plan: ScenePlan) -> str:
    """把 SceneProgram 编译为不含动态执行的 Manim Python。"""

    if len({item.element_id for item in program.objects}) != len(program.objects):
        raise SceneProgramCompileError("SceneProgram 包含重复 element_id")
    exported_ids = {
        item.element_id for item in program.objects if item.exported and not item.removed
    }
    required_ids = {
        item.element_id
        for item in [*scene_plan.inherited_elements, *scene_plan.new_elements]
        if item.required
        and item.element_id not in {x.element_id for x in scene_plan.elements_to_remove}
    }
    if exported_ids != required_ids:
        missing = required_ids - exported_ids
        extra = exported_ids - required_ids
        raise SceneProgramCompileError(
            "SceneProgram 导出集合不一致: "
            + (f"缺少 {sorted(missing)} " if missing else "")
            + (f"多出 {sorted(extra)}" if extra else "")
        )
    needs_numpy = any(item.kind == "Surface" for item in program.objects)
    lines = ["from manim import *"]
    if needs_numpy:
        lines.append("import numpy as np")
    lines.extend(
        [
            "",
            f"class {program.class_name}({program.parent}):",
            "    def construct(self):",
            '        COLORS = {"primary": BLUE, "secondary": GREEN, "highlight": YELLOW}',
        ]
    )
    removed_objects = [item for item in program.objects if item.removed]
    exported_objects = [item for item in program.objects if item.exported and not item.removed]
    variables: set[str] = set()
    for item in removed_objects:
        lines.append(f"        {item.variable_name} = {_object_expression(item, variables)}")
        variables.add(item.variable_name)
    lines.append("        # KD1_CONTINUITY_EXPORT_BEGIN")
    for item in exported_objects:
        lines.append(f"        # element_id: {item.element_id}")
        lines.append(f"        {item.variable_name} = {_object_expression(item, variables)}")
        variables.add(item.variable_name)
    lines.append("        # KD1_CONTINUITY_EXPORT_END")
    lines.extend(
        [
            "",
            f"        title = Text({_quote(scene_plan.title[:120])}, font_size=32)",
            f"        summary = Text({_quote(program.summary[:240])}, font_size=24)",
            "        summary.next_to(title, DOWN, buff=0.5)",
            "        self.play(FadeIn(title), FadeIn(summary), run_time=0.5)",
        ]
    )
    object_by_id = {item.element_id: item for item in program.objects}
    for animation in program.animations:
        if animation.operation == "wait":
            lines.append(f"        self.wait({animation.run_time:g})")
            continue
        source_vars = [object_by_id[item].variable_name for item in animation.source_ids]
        if not source_vars:
            raise SceneProgramCompileError(f"动画 {animation.operation} 缺少 source")
        if animation.operation == "add":
            lines.append(f"        self.add({', '.join(source_vars)})")
        elif animation.operation == "fade_out":
            lines.append(
                f"        self.play(FadeOut({', '.join(source_vars)}), run_time={animation.run_time:g})"
            )
        elif animation.operation in {"fade_in", "create", "write"}:
            constructor = {"fade_in": "FadeIn", "create": "Create", "write": "Write"}[
                animation.operation
            ]
            lines.append(
                f"        self.play({', '.join(f'{constructor}({name})' for name in source_vars)}, run_time={animation.run_time:g})"
            )
        else:
            raise SceneProgramCompileError(f"后备编译器暂不支持动画: {animation.operation}")
    lines.append(f"        self.wait({program.wait_seconds:g})")
    return "\n".join(lines) + "\n"


__all__ = [
    "ProgramAnimation",
    "ProgramObject",
    "SceneProgram",
    "SceneProgramCompileError",
    "build_scene_program_from_contract",
    "compile_scene_program",
]
