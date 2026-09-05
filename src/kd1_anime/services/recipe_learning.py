"""匿名配方学习服务。"""

from __future__ import annotations

from pathlib import Path

from kd1_anime.rag.recipes import RecipeRecord, RecipeStore


class RecipeLearningService:
    """把配方存储从 Orchestrator 中隔离出来。"""

    def __init__(self, store: RecipeStore | None = None) -> None:
        self.store = store or RecipeStore()

    def save(
        self,
        code: str,
        *,
        renderer: str,
        semantic_intent: str = "",
        object_kinds: list[str] | tuple[str, ...] = (),
        capabilities: list[str] | tuple[str, ...] = (),
        verification: str = "rendered",
    ) -> tuple[RecipeRecord, Path, bool]:
        return self.store.save(
            code,
            renderer=renderer,
            semantic_intent=semantic_intent,
            object_kinds=object_kinds,
            capabilities=capabilities,
            verification=verification,
        )


__all__ = ["RecipeLearningService"]
