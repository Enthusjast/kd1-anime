"""评估指标及可持久化结果。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from kd1_anime.run_store import atomic_write_json


class EvalMetric(str, Enum):
    CODE_SYNTAX = "code_syntax"
    CODE_SECURITY = "code_security"
    CODE_COMPLEXITY = "code_complexity"
    CODE_STYLE = "code_style"
    VISUAL_MATH_ACCURACY = "visual_math_accuracy"
    VISUAL_RELEVANCE = "visual_relevance"
    VISUAL_QUALITY = "visual_quality"
    VISUAL_CONSISTENCY = "visual_consistency"
    ELEMENT_LAYOUT = "element_layout"
    RENDER_TIME = "render_time"
    SUCCESS_RATE = "success_rate"
    RETRY_COUNT = "retry_count"


class ScoreLevel(str, Enum):
    VERY_POOR = "very_poor"
    BELOW_AVG = "below_average"
    ACCEPTABLE = "acceptable"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass
class QualityScore:
    metric: EvalMetric
    score: int
    justification: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise TypeError(f"Score must be an integer, got {type(self.score).__name__}")
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be between 1 and 5, got {self.score}")

    @property
    def level(self) -> ScoreLevel:
        return {
            1: ScoreLevel.VERY_POOR,
            2: ScoreLevel.BELOW_AVG,
            3: ScoreLevel.ACCEPTABLE,
            4: ScoreLevel.GOOD,
            5: ScoreLevel.EXCELLENT,
        }[self.score]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "score": self.score,
            "level": self.level.value,
            "justification": self.justification,
            "details": self.details,
        }


@dataclass
class EvalResult:
    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scores: list[QualityScore] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    overall_score: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._recalculate_overall()

    def add_score(self, score: QualityScore) -> None:
        self.scores.append(score)
        self._recalculate_overall()

    def add_error(self, category: str, message: str) -> None:
        self.errors[category] = message

    def _recalculate_overall(self) -> None:
        if not self.scores:
            self.overall_score = None
            return
        product = 1.0
        for score in self.scores:
            product *= score.score
        self.overall_score = product ** (1.0 / len(self.scores))

    def get_score(self, metric: EvalMetric) -> QualityScore | None:
        return next((score for score in self.scores if score.metric == metric), None)

    def get_metric_average(self, metric: EvalMetric) -> float | None:
        values = [score.score for score in self.scores if score.metric == metric]
        return sum(values) / len(values) if values else None

    def get_scores_by_category(self, category: str) -> list[QualityScore]:
        return [score for score in self.scores if score.metric.value.startswith(category)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": (
                round(self.overall_score, 2) if self.overall_score is not None else None
            ),
            "summary": self.summary,
            "scores": [score.to_dict() for score in self.scores],
            "metadata": self.metadata,
            "errors": self.errors,
        }

    def save(self, output_path: Path) -> None:
        atomic_write_json(output_path, self.to_dict())

    @classmethod
    def load(cls, input_path: Path) -> EvalResult:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        result = cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            summary=data.get("summary", ""),
            metadata=data.get("metadata", {}),
            errors=data.get("errors", {}),
        )
        for item in data.get("scores", []):
            result.add_score(
                QualityScore(
                    metric=EvalMetric(item["metric"]),
                    score=item["score"],
                    justification=item.get("justification", ""),
                    details=item.get("details", {}),
                )
            )
        # 不信任文件中的 overall_score，始终由实际指标重新计算。
        result._recalculate_overall()
        return result


@dataclass
class ComparisonResult:
    baseline_run_id: str
    current_run_id: str
    baseline_result: EvalResult
    current_result: EvalResult
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)

    @property
    def score_diff(self) -> float | None:
        baseline = self.baseline_result.overall_score
        current = self.current_result.overall_score
        if baseline is None or current is None:
            return None
        return current - baseline

    def to_dict(self) -> dict[str, Any]:
        baseline = self.baseline_result.overall_score
        current = self.current_result.overall_score
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "baseline_score": round(baseline, 2) if baseline is not None else None,
            "current_score": round(current, 2) if current is not None else None,
            "score_diff": round(self.score_diff, 2) if self.score_diff is not None else None,
            "improvements": self.improvements,
            "regressions": self.regressions,
        }
