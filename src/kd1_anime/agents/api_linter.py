"""ManimCE API 级静态检查。

该模块只分析 AST，不执行生成代码。它补充 validator 的安全边界和
lifecycle 的 active 状态模拟，专门发现高频 Manim API 误用及性能风险。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from kd1_anime.agents.planner import ScenePlan

Renderer = Literal["cairo", "opengl"]

_DEPRECATED_CALLS = {
    "ShowCreation": "Create",
    "ShowCreationThenFadeOut": "Create + FadeOut",
    "TextMobject": "Text",
    "TexMobject": "MathTex",
}
_DEPRECATED_METHODS = {
    "setColor": "set_color",
    "moveToEdge": "to_edge",
    "moveTo": "move_to",
}
_THREE_D_CONSTRUCTORS = {
    "Arrow3D",
    "Cone",
    "Cube",
    "Cylinder",
    "Line3D",
    "ParametricSurface",
    "Polyhedron",
    "Sphere",
    "Surface",
    "Tetrahedron",
    "ThreeDAxes",
    "Torus",
}


@dataclass(frozen=True, slots=True)
class ApiLintResult:
    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _scene_bases(tree: ast.AST) -> set[str]:
    bases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {
                "Scene",
                "ThreeDScene",
                "MovingCameraScene",
            }:
                bases.add(base.id)
    return bases


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def lint_manim_api(
    code: str,
    *,
    renderer: Renderer | None = None,
    scene_plan: ScenePlan | None = None,
) -> ApiLintResult:
    """检查常见的 ManimCE API 错误，无法确定的情况只给 warning。"""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ApiLintResult(True)

    errors: list[str] = []
    warnings: list[str] = []
    bases = _scene_bases(tree)
    effective_renderer = renderer or "cairo"
    has_updater = False
    has_updater_cleanup = False
    risk_text = ""
    if scene_plan is not None:
        risk_text = " ".join(
            (
                scene_plan.math_concept,
                scene_plan.computation,
                scene_plan.visual_design,
                *scene_plan.visual_flow,
            )
        ).lower()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name in _DEPRECATED_CALLS:
            errors.append(
                f"第 {node.lineno} 行使用已废弃 API {call_name}，应改为 {_DEPRECATED_CALLS[call_name]}"
            )
        if call_name in _THREE_D_CONSTRUCTORS and "ThreeDScene" not in bases:
            errors.append(f"第 {node.lineno} 行使用 {call_name}，但场景没有继承 ThreeDScene")
        if call_name in {"Surface", "ParametricSurface"} and not any(
            keyword.arg == "resolution" for keyword in node.keywords
        ):
            warnings.append(
                f"第 {node.lineno} 行 {call_name} 未显式设置 resolution，复杂曲面可能导致渲染过慢"
            )
        if call_name == "LaggedStart" and not any(
            keyword.arg == "run_time" for keyword in node.keywords
        ):
            warnings.append(
                f"第 {node.lineno} 行 LaggedStart 未设置 run_time，建议显式控制动画时长"
            )
        if call_name in {"always_redraw", "add_updater"}:
            has_updater = True
        if call_name == "clear_updaters":
            has_updater_cleanup = True
        if isinstance(node.func, ast.Attribute) and node.func.attr in _DEPRECATED_METHODS:
            errors.append(
                f"第 {node.lineno} 行使用已废弃方法 {node.func.attr}，应改为 "
                f"{_DEPRECATED_METHODS[node.func.attr]}"
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "plot"
            and not any(keyword.arg == "x_range" for keyword in node.keywords)
        ):
            warnings.append(
                f"第 {node.lineno} 行 Axes.plot 未显式设置 x_range，请确认定义域和奇点处理"
            )
        if call_name == "TransformMatchingTex":
            warnings.append(
                f"第 {node.lineno} 行 TransformMatchingTex 需确认两侧公式存在可匹配子串"
            )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "frame"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "camera"
        ):
            if effective_renderer == "opengl":
                errors.append(f"第 {node.lineno} 行 OpenGL 不支持 self.camera.frame")
            elif "MovingCameraScene" not in bases:
                errors.append(f"第 {node.lineno} 行普通 Scene 不支持 self.camera.frame")

    if has_updater and not has_updater_cleanup:
        warnings.append(
            "场景使用 updater/always_redraw 但未发现 clear_updaters()，请确认不会跨阶段残留"
        )

    if any(marker in risk_text for marker in ("1/x", "1 / x", "奇点", "间断", "除法")):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "plot"
                and not any(keyword.arg == "x_range" for keyword in node.keywords)
            ):
                warnings.append("计划涉及可能存在定义域限制的函数，建议拆分曲线 x_range")
                break

    return ApiLintResult(
        is_valid=not tuple(dict.fromkeys(errors)),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["ApiLintResult", "lint_manim_api"]
