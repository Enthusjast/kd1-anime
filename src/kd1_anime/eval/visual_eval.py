"""使用多模态 LLM 对最终视频关键帧进行视觉质量评估。"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.agents.base import BaseAgent
from kd1_anime.config import settings
from kd1_anime.eval.metrics import EvalMetric, QualityScore
from kd1_anime.eval.prompts import VISUAL_EVAL_PROMPT

MAX_FRAME_COUNT = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024


class _Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(strict=True, ge=1, le=5)
    comprehensive_evaluation: str = ""


class _Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_relevance: _Dimension
    visual_quality: _Dimension
    visual_consistency: _Dimension
    element_layout: _Dimension


class _VisualPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_analysis: str = ""
    evaluation: _Evaluation


@dataclass
class VisualAnalysisResult:
    overall_analysis: str
    visual_relevance: dict[str, Any]
    visual_quality: dict[str, Any]
    visual_consistency: dict[str, Any]
    element_layout: dict[str, Any]
    raw_response: str = ""


class VisualEvaluator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EVAL_VISUAL_MODEL or settings.LLM_MODEL
        self._agent: BaseAgent | None = None

    @property
    def agent(self) -> BaseAgent:
        if self._agent is None:
            self._agent = BaseAgent()
            self._agent.model = self.model_name
        return self._agent

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                raise ValueError(f"Image too large: {path} ({size} bytes > {MAX_IMAGE_BYTES})")
            with path.open("rb") as handle:
                payload = handle.read(MAX_IMAGE_BYTES + 1)
        except OSError as exc:
            raise OSError(f"读取图片失败: {path}") from exc
        if not payload:
            raise ValueError(f"Image is empty: {path}")
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image too large: {path} ({len(payload)} bytes > {MAX_IMAGE_BYTES})")
        return base64.b64encode(payload).decode("ascii")

    @staticmethod
    def _image_content(path: Path) -> dict[str, Any]:
        mime, _ = mimetypes.guess_type(path.name)
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(f"不支持的关键帧格式: {path.suffix}")
        encoded = VisualEvaluator.encode_image(path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    def evaluate_image(
        self,
        image_path: str | Path,
        description: str = "Mathematical animation",
    ) -> VisualAnalysisResult:
        return self.evaluate_video_frames([Path(image_path)], description)

    def evaluate_video_frames(
        self,
        frame_paths: list[Path],
        description: str = "Mathematical animation",
    ) -> VisualAnalysisResult:
        if not frame_paths:
            raise ValueError("No frame paths provided")
        if len(frame_paths) > MAX_FRAME_COUNT:
            raise ValueError(f"关键帧数量不能超过 {MAX_FRAME_COUNT}")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": VISUAL_EVAL_PROMPT.format(description=description),
            }
        ]
        content.extend(self._image_content(path) for path in frame_paths)
        response = self.agent.call_llm(
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=2000,
            json_mode=True,
            stream=False,
        )
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> VisualAnalysisResult:
        payload = BaseAgent._extract_json(response)
        parsed = _VisualPayload.model_validate_json(payload)
        evaluation = parsed.evaluation
        return VisualAnalysisResult(
            overall_analysis=parsed.overall_analysis,
            visual_relevance=evaluation.visual_relevance.model_dump(),
            visual_quality=evaluation.visual_quality.model_dump(),
            visual_consistency=evaluation.visual_consistency.model_dump(),
            element_layout=evaluation.element_layout.model_dump(),
            raw_response=response,
        )

    def evaluate_frames(
        self,
        frame_paths: list[Path],
        description: str = "",
    ) -> list[QualityScore]:
        result = self.evaluate_video_frames(frame_paths, description)
        dimensions = (
            (EvalMetric.VISUAL_RELEVANCE, result.visual_relevance),
            (EvalMetric.VISUAL_QUALITY, result.visual_quality),
            (EvalMetric.VISUAL_CONSISTENCY, result.visual_consistency),
            (EvalMetric.ELEMENT_LAYOUT, result.element_layout),
        )
        return [
            QualityScore(
                metric=metric,
                score=data["score"],
                justification=data.get("comprehensive_evaluation", ""),
                details=data,
            )
            for metric, data in dimensions
        ]

    def evaluate(self, image_path: str | Path, description: str = "") -> list[QualityScore]:
        return self.evaluate_frames([Path(image_path)], description)
