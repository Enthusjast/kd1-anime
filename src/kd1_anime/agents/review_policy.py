"""按场景风险分配有限代码审查预算。"""

from __future__ import annotations

from dataclasses import dataclass

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.risk import RiskLevel, assess_scene_risk
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.config import GenerationMode


@dataclass(frozen=True, slots=True)
class ReviewBudget:
    risk_level: RiskLevel
    max_rounds: int | None
    deterministic_checks_required: bool = True


@dataclass(frozen=True, slots=True)
class ReviewModePolicy:
    """统一的 Review/修复预算策略。

    ``None`` 表示不使用固定轮数上限；停滞检测仍由状态机负责。
    """

    mode: GenerationMode
    strict_cap: int = 8

    @property
    def relaxed(self) -> bool:
        return self.mode == "relaxed"

    def limit(self, configured: int | None) -> int | None:
        if self.relaxed:
            return None
        value = self.strict_cap if configured is None else int(configured)
        return max(1, min(self.strict_cap, value))


def review_mode_policy(mode: GenerationMode) -> ReviewModePolicy:
    return ReviewModePolicy(mode=mode)


def review_budget(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None,
    *,
    global_max_rounds: int,
    low_risk_max_rounds: int = 2,
    generation_mode: GenerationMode = "strict",
) -> ReviewBudget:
    """返回 LLM 审查上限；确定性校验不受该预算影响。"""

    risk = assess_scene_risk(scene_plan, technical_spec)
    if generation_mode == "relaxed":
        return ReviewBudget(risk_level=risk.level, max_rounds=None)
    limit = low_risk_max_rounds if risk.level == "low" else global_max_rounds
    return ReviewBudget(
        risk_level=risk.level,
        max_rounds=max(1, min(8, int(limit))),
    )


__all__ = ["ReviewBudget", "ReviewModePolicy", "review_budget", "review_mode_policy"]
