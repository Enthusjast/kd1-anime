"""独立视觉模型调用的组合服务。"""

from __future__ import annotations


class VisualEvaluationService:
    """隔离视觉评估调用，避免 FSM 直接依赖客户端细节。"""

    @staticmethod
    def evaluate_frames(
        evaluator,
        samples,
        description: str,
        *,
        scene_context: str,
        scope: str,
    ):
        if evaluator is None:
            raise RuntimeError("视觉评估器未初始化")
        return evaluator.evaluate_video_frames(
            samples,
            description,
            scene_context=scene_context,
            scope=scope,
        )


__all__ = ["VisualEvaluationService"]
