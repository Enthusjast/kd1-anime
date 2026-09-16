"""流水线组合服务。

这些服务只持有一个清晰的外部边界；Orchestrator 负责 FSM 和状态协调，
服务负责渲染、恢复、计划辅助、视觉调用及配方学习等可替换能力。
"""

from kd1_anime.services.planning import PlanningService
from kd1_anime.services.recipe_learning import RecipeLearningService
from kd1_anime.services.recovery import RecoveryService
from kd1_anime.services.rendering import RenderingService
from kd1_anime.services.visual_evaluation import VisualEvaluationService

__all__ = [
    "PlanningService",
    "RecipeLearningService",
    "RecoveryService",
    "RenderingService",
    "VisualEvaluationService",
]
