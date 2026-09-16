"""计划阶段的纯辅助逻辑。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


class PlanningService:
    """集中处理不依赖 FSM 的计划指纹和时长计算。"""

    @staticmethod
    def cycle_signature(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def expected_duration(
        durations: Sequence[float],
        transition_duration: float,
    ) -> float:
        values = [float(duration) for duration in durations]
        if not values:
            return 0.0
        transition = min(
            max(0.0, float(transition_duration)),
            max(0.0, min(values) / 2) if len(values) > 1 else 0.0,
        )
        return max(0.0, sum(values) - transition * max(0, len(values) - 1))


__all__ = ["PlanningService"]
