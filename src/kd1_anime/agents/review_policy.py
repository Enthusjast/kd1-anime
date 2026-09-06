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

    def prompt_guidance(self) -> str:
        """返回给审查模型的模式约束。

        relaxed 并不是关闭确定性校验，而是把 LLM 的主观判断降级为
        warning。把这条规则显式放进 prompt，可以减少模型把布局偏好、
        风格差异或无法核验的推测包装成 major finding，避免无效重写。
        """

        if self.relaxed:
            return (
                "## 当前生成模式：relaxed\n"
                "本模式优先保证生成速度和可渲染成功率。AST、Manim API、数学/几何"
                "确定性检查、动画生命周期、连续性导出合同和能力合同仍然有效，"
                "不得忽略真实的运行时错误或明确的数学错误。除此之外，不要因为个人"
                "实现偏好、布局建议、风格差异、一般节奏建议或无法从输入直接核验的"
                "推测返回 major；这类意见应放入 warnings。若确定性检查通过且没有"
                "高置信度、带源码/合同证据的核心错误，应返回 is_valid=true。"
            )
        return (
            "## 当前生成模式：strict\n"
            "严格执行确定性检查，并检查核心数学、运行时、动画生命周期、连续性合同和安全边界；"
            "只有能够从当前输入直接核验的问题才能作为阻断项。"
        )


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


def review_mode_guidance(mode: GenerationMode) -> str:
    """返回适合嵌入 Planner/Reviewer system prompt 的统一模式说明。"""

    return review_mode_policy(mode).prompt_guidance()


__all__ = [
    "ReviewBudget",
    "ReviewModePolicy",
    "review_budget",
    "review_mode_guidance",
    "review_mode_policy",
]
