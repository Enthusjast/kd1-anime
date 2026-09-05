"""用于约束 Coder 输出结构的轻量场景模板。

模板只负责提供稳定的文件骨架和 renderer/API 边界，不替 Planner 创作
动画内容。具体的对象、数学关系和动画事件仍由 ScenePlan 与 TechnicalSpec
决定。
"""

from __future__ import annotations

import json
import re
from typing import Literal

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec

SceneTemplateKind = Literal[
    "formula",
    "graph",
    "geometry",
    "surface",
    "moving_camera",
    "updater",
    "generic",
]


def select_scene_template(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
    *,
    renderer: str | None = None,
) -> SceneTemplateKind:
    """根据已经确定的计划选择最小的代码骨架。

    只读取结构化合同和计划文本，不调用 LLM；无法确定时使用 generic，
    从而保持与旧版 Coder 的兼容性。
    """

    object_text = ""
    if technical_spec is not None:
        object_text = " ".join(
            f"{item.constructor} {item.visual_role} {item.initial_state} {item.final_state}"
            for item in technical_spec.objects
        )
    text = " ".join(
        (
            scene_plan.title,
            scene_plan.purpose,
            scene_plan.math_concept,
            scene_plan.visual_design,
            scene_plan.camera_movement,
            *scene_plan.visual_flow,
            scene_plan.computation,
            object_text,
        )
    ).lower()

    if any(
        marker in text
        for marker in (
            "threedscene",
            "surface(",
            "parametric_surface",
            "三维",
            "曲面",
            "立体",
        )
    ):
        return "surface"
    if any(
        marker in text
        for marker in ("valuetracker", "always_redraw", "add_updater", "updater", "动态参数")
    ):
        return "updater"
    if any(
        marker in text
        for marker in (
            "movingcamerascene",
            "camera.frame",
            "相机缩放",
            "镜头推近",
            "镜头平移",
            "zoom",
            "pan",
        )
    ):
        return "moving_camera" if renderer != "opengl" else "generic"
    if any(
        marker in text
        for marker in (
            "axes(",
            "numberplane",
            "coordinate",
            "坐标系",
            "函数图像",
            "曲线",
            "plot",
        )
    ):
        return "graph"
    if scene_plan.geometry_specs or any(
        marker in text
        for marker in ("面积", "三角形", "正方形", "矩形", "圆形", "polygon", "geometry")
    ):
        return "geometry"
    if any(
        marker in text
        for marker in (
            "mathtex",
            "latex",
            "公式",
            "等式",
            "推导",
            "equation",
            "formula",
            "tex(",
        )
    ):
        return "formula"
    return "generic"


def _base_template(scene_id: int, parent: str, *, marker: str, example: str) -> str:
    class_name = f"Scene{scene_id}"
    return f"""```python
from manim import *

class {class_name}({parent}):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{{ctex}}")
        config.tex_template = tex_template
        COLORS = {{"primary": BLUE, "secondary": GREEN, "highlight": YELLOW}}

        # KD1_CONTINUITY_EXPORT_BEGIN
        # {marker}: 在此定义合同要求的 required=true 元素；禁止保留 TODO
        # KD1_CONTINUITY_EXPORT_END

        # 下面是该模板的最小参考实现；边界对象必须按合同移动到 marker 内。
{example}
        # 必须在本方法中完成 self.add/self.play/self.wait。
```"""


def build_scene_template(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
    *,
    renderer: str | None = None,
) -> str:
    """返回注入 Coder user prompt 的参考骨架。"""

    kind = select_scene_template(scene_plan, technical_spec, renderer=renderer)
    parent = {
        "surface": "ThreeDScene",
        "moving_camera": "MovingCameraScene",
    }.get(kind, "Scene")
    marker = {
        "formula": "formula_elements",
        "graph": "graph_elements",
        "geometry": "geometry_elements",
        "surface": "surface_elements",
        "moving_camera": "camera_elements",
        "updater": "dynamic_elements",
        "generic": "scene_elements",
    }[kind]
    examples = {
        "formula": (
            '        formula = MathTex(r"a^2+b^2=c^2", tex_template=tex_template)\n'
            "        # KD1_ANIMATION_EVENT: show_formula\n"
            "        self.play(Write(formula), run_time=1)"
        ),
        "graph": (
            "        axes = Axes(x_range=[-4, 4, 1], y_range=[-2, 8, 2])\n"
            '        graph = axes.plot(lambda x: x**2, x_range=[-2.5, 2.5], color=COLORS["primary"])\n'
            "        # KD1_ANIMATION_EVENT: draw_graph\n"
            "        self.play(Create(axes), Create(graph), run_time=2)"
        ),
        "geometry": (
            '        shape = Polygon(LEFT * 2, RIGHT * 2, UP * 2, color=COLORS["primary"])\n'
            "        # KD1_ANIMATION_EVENT: show_geometry\n"
            "        self.play(Create(shape), run_time=1)"
        ),
        "surface": (
            "        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)\n"
            "        axes = ThreeDAxes()\n"
            "        surface = Surface(lambda u, v: axes.c2p(u, v, u**2 + v**2), u_range=[-2, 2], v_range=[-2, 2], resolution=(12, 12))\n"
            "        # KD1_ANIMATION_EVENT: show_surface\n"
            "        self.play(Create(axes), Create(surface), run_time=2)"
        ),
        "moving_camera": (
            '        focus = Circle(color=COLORS["primary"])\n'
            "        self.add(focus)\n"
            "        self.camera.frame.save_state()\n"
            "        # KD1_ANIMATION_EVENT: move_camera\n"
            "        self.play(self.camera.frame.animate.scale(0.6).move_to(focus), run_time=1.5)\n"
            "        # KD1_ANIMATION_EVENT: restore_camera\n"
            "        self.play(Restore(self.camera.frame), run_time=1)"
        ),
        "updater": (
            "        tracker = ValueTracker(0)\n"
            '        dot = always_redraw(lambda: Dot(RIGHT * tracker.get_value(), color=COLORS["highlight"]))\n'
            "        self.add(dot)\n"
            "        # KD1_ANIMATION_EVENT: update_tracker\n"
            "        self.play(tracker.animate.set_value(3), run_time=2, rate_func=smooth)\n"
            "        dot.clear_updaters()"
        ),
        "generic": "        # 根据 TechnicalSpec 定义对象并实现动画事件",
    }
    template = _base_template(
        scene_plan.scene_id,
        parent,
        marker=marker,
        example=examples[kind],
    )
    return (
        f"模板类型: {kind}\n"
        "下面是稳定的文件骨架，不是可直接提交的最终代码。必须替换所有示意注释，"
        "严格按照当前 ScenePlan、TechnicalSpec、连续性导出合同填写；不得增加第二个 Scene 类。\n"
        f"{template}"
    )


def build_safe_scene_code(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
) -> str:
    """构造不依赖 LLM 的最小可渲染代码。

    该代码不是正常创作路径，而是 Coder 输出为空、截断或反复无效时的
    最后保险。它只展示场景标题和合同中声明的元素标签，不猜测坐标、几何
    碎片或数学推导；结果仍由调用方执行全部确定性校验和 Code Review。
    """

    kind = select_scene_template(scene_plan, technical_spec)
    parent = "ThreeDScene" if kind == "surface" else "Scene"
    objects = {item.element_id: item for item in (technical_spec.objects if technical_spec else ())}
    removed_ids = {item.element_id for item in scene_plan.elements_to_remove}
    declared = [*scene_plan.inherited_elements, *scene_plan.new_elements]
    required = [item for item in declared if item.required and item.element_id not in removed_ids]
    removed = [item for item in scene_plan.elements_to_remove if item.element_id in objects]

    def variable_name(item: object) -> str:
        element_id = str(getattr(item, "element_id", "element"))
        technical = objects.get(element_id)
        candidate = str(getattr(technical, "variable_name", "") or "")
        candidate = candidate or str(getattr(item, "variable_name", "") or "")
        candidate = candidate or re.sub(r"[^A-Za-z0-9_]", "_", element_id)
        if candidate and (candidate[0].isalpha() or candidate[0] == "_"):
            return candidate
        return "element_" + candidate

    def quote(value: str) -> str:
        return json.dumps(str(value or ""), ensure_ascii=False)

    lines = [
        "from manim import *",
        "",
        f"class Scene{scene_plan.scene_id}({parent}):",
        "    def construct(self):",
        '        COLORS = {"primary": BLUE, "secondary": GREEN, "highlight": YELLOW}',
    ]
    # 被移除对象需要在 marker 外定义，并先加入场景再 FadeOut；这样既不
    # 进入边界导出，也满足生命周期检查的 source/active 约束。
    for item in removed:
        variable = variable_name(item)
        lines.append(
            f"        {variable} = Text({quote(item.role or item.element_id)}, "
            'color=COLORS["primary"])'
        )
    lines.extend(["", "        # KD1_CONTINUITY_EXPORT_BEGIN"])
    for item in required:
        variable = variable_name(item)
        lines.append(f"        # element_id: {item.element_id}")
        lines.append(
            f"        {variable} = Text({quote(item.role or item.element_id)}, "
            'color=COLORS["primary"])'
        )
    lines.extend(
        [
            "        # KD1_CONTINUITY_EXPORT_END",
            "",
            f"        title = Text({quote(scene_plan.title[:120])}, font_size=32)",
            f"        summary = Text({quote((scene_plan.math_concept + '：' + scene_plan.computation)[:240])}, font_size=24)",
            "        summary.next_to(title, DOWN, buff=0.5)",
            "        # KD1_ANIMATION_EVENT: __auto_aux_title",
            "        self.play(FadeIn(title), run_time=0.5)",
            "        # KD1_ANIMATION_EVENT: __auto_aux_summary",
            "        self.play(FadeIn(summary), run_time=0.5)",
        ]
    )
    for index, item in enumerate(removed):
        variable = variable_name(item)
        lines.extend(
            [
                f"        {variable}.move_to(DOWN * {index + 1})",
                f"        self.add({variable})",
                f"        # KD1_ANIMATION_EVENT: __auto_remove_{variable}",
                f"        self.play(FadeOut({variable}), run_time=0.3)",
            ]
        )
    for index, item in enumerate(required):
        variable = variable_name(item)
        lines.append(f"        {variable}.move_to(DOWN * {index + 1})")
        technical = objects.get(item.element_id)
        if bool(getattr(technical, "initially_active", False)):
            lines.append(f"        self.add({variable})")
        else:
            lines.append(f"        # KD1_ANIMATION_EVENT: __auto_introduce_{variable}")
            lines.append(f"        self.play(FadeIn({variable}), run_time=0.3)")
    lines.append("        self.wait(1)")
    return "\n".join(lines) + "\n"


__all__ = [
    "SceneTemplateKind",
    "build_safe_scene_code",
    "build_scene_template",
    "select_scene_template",
]
