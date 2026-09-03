"""用于约束 Coder 输出结构的轻量场景模板。

模板只负责提供稳定的文件骨架和 renderer/API 边界，不替 Planner 创作
动画内容。具体的对象、数学关系和动画事件仍由 ScenePlan 与 TechnicalSpec
决定。
"""

from __future__ import annotations

from typing import Literal

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec

SceneTemplateKind = Literal["formula", "graph", "geometry", "surface", "generic"]


def select_scene_template(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
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


def _base_template(scene_id: int, parent: str, *, marker: str) -> str:
    class_name = f"Scene{scene_id}"
    return f'''```python
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

        # 用 TechnicalSpec 中声明的真实对象和动画替换这里的示意注释。
        # 必须在本方法中完成 self.add/self.play/self.wait。
```'''


def build_scene_template(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
) -> str:
    """返回注入 Coder user prompt 的参考骨架。"""

    kind = select_scene_template(scene_plan, technical_spec)
    parent = "ThreeDScene" if kind == "surface" else "Scene"
    marker = {
        "formula": "formula_elements",
        "graph": "graph_elements",
        "geometry": "geometry_elements",
        "surface": "surface_elements",
        "generic": "scene_elements",
    }[kind]
    template = _base_template(scene_plan.scene_id, parent, marker=marker)
    return (
        f"模板类型: {kind}\n"
        "下面是稳定的文件骨架，不是可直接提交的最终代码。必须替换所有示意注释，"
        "严格按照当前 ScenePlan、TechnicalSpec、连续性导出合同填写；不得增加第二个 Scene 类。\n"
        f"{template}"
    )


__all__ = ["SceneTemplateKind", "build_scene_template", "select_scene_template"]
