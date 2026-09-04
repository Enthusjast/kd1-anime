"""按场景风险分配有限代码审查预算。"""

from __future__ import annotations

from dataclasses import dataclass

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.risk import RiskLevel, assess_scene_risk
from kd1_anime.agents.technical_planner import TechnicalSpec


@dataclass(frozen=True, slots=True)
class ReviewBudget:
    risk_level: RiskLevel
    max_rounds: int
    deterministic_checks_required: bool = True


def review_budget(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None,
    *,
    global_max_rounds: int,
    low_risk_max_rounds: int = 2,
) -> ReviewBudget:
    """返回 LLM 审查上限；确定性校验不受该预算影响。"""

    risk = assess_scene_risk(scene_plan, technical_spec)
    limit = low_risk_max_rounds if risk.level == "low" else global_max_rounds
    return ReviewBudget(
        risk_level=risk.level,
        max_rounds=max(1, min(int(global_max_rounds), int(limit))),
    )


__all__ = ["ReviewBudget", "review_budget"]
