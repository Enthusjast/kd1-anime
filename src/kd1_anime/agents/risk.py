"""场景复杂度评估，用于选择有限的代码候选策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec

RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class SceneRisk:
    """一个场景的可解释风险分数。"""

    level: RiskLevel
    score: int
    reasons: tuple[str, ...] = ()


def assess_scene_risk(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
) -> SceneRisk:
    """按已知结构特征估计实现风险，不调用 LLM、不执行生成代码。"""

    score = 0
    reasons: list[str] = []
    text = " ".join(
        (
            scene_plan.visual_design,
            scene_plan.camera_movement,
            scene_plan.computation,
            *scene_plan.visual_flow,
            *scene_plan.key_moments,
        )
    ).lower()
    if any(token in text for token in ("3d", "三维", "曲面", "切平面", "surface", "threedscene")):
        score += 4
        reasons.append("三维/曲面或切平面")
    if any(token in text for token in ("updater", "always_redraw", "动态轨迹", "逐帧")):
        score += 3
        reasons.append("逐帧更新或 updater")
    if any(token in text for token in ("切割", "拼接", "碎片", "面积守恒", "旋转")):
        score += 3
        reasons.append("几何碎片或面积变换")
    if len(scene_plan.geometry_specs) >= 3:
        score += 2
        reasons.append("多个几何规格")
    if len(scene_plan.math_claims) >= 5:
        score += 1
        reasons.append("数学断言较多")
    if len(scene_plan.timeline) >= 10 or len(scene_plan.visual_flow) >= 10:
        score += 1
        reasons.append("时间线事件较多")
    if scene_plan.inherited_elements:
        score += 1
        reasons.append("存在跨场景继承")
    if technical_spec is not None:
        if technical_spec.renderer == "opengl":
            score += 2
            reasons.append("OpenGL 渲染")
        if len(technical_spec.objects) >= 8:
            score += 2
            reasons.append("技术对象较多")
        if len(technical_spec.animations) >= 12:
            score += 2
            reasons.append("动画事件较多")
        if technical_spec.latex.required:
            score += 1
            reasons.append("需要 XeLaTeX")

    if score >= 7:
        level: RiskLevel = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"
    return SceneRisk(level=level, score=score, reasons=tuple(reasons))


__all__ = ["RiskLevel", "SceneRisk", "assess_scene_risk"]
