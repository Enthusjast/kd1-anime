"""使用独立多模态 LLM 对 Manim 视频关键帧进行视觉质量评估。"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.agents.base import BaseAgent
from kd1_anime.config import settings
from kd1_anime.eval.metrics import EvalMetric, QualityScore
from kd1_anime.eval.prompts import VISUAL_EVAL_PROMPT, VISUAL_EVAL_SYSTEM_PROMPT
from kd1_anime.rendering import sha256_file

MAX_FRAME_COUNT = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FrameSample(BaseModel):
    """关键帧及其可信提取元数据。"""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(pattern=r"^F\d{2}$")
    path: Path
    timestamp_seconds: float | None = Field(default=None, ge=0)
    image_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    role: Literal[
        "opening",
        "first_math_state",
        "middle",
        "conclusion",
        "ending",
        "transition_boundary",
        "content",
    ] = "content"


class _Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(strict=True, ge=1, le=5)
    comprehensive_evaluation: str = Field(default="", max_length=4_000)


class _Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mathematical_accuracy: _Dimension
    visual_relevance: _Dimension
    visual_quality: _Dimension
    visual_consistency: _Dimension
    element_layout: _Dimension


class VisualIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "mathematics",
        "relevance",
        "readability",
        "layout",
        "clipping",
        "overlap",
        "contrast",
        "consistency",
        "other",
    ]
    severity: Literal["info", "minor", "major"]
    frame_ids: list[str] = Field(default_factory=list, max_length=MAX_FRAME_COUNT)
    evidence: str = Field(min_length=1, max_length=2_000)
    recommendation: str = Field(min_length=1, max_length=2_000)


class _VisualPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_analysis: str = Field(default="", max_length=4_000)
    evaluation: _Evaluation
    issues: list[VisualIssue] = Field(default_factory=list, max_length=20)


@dataclass(slots=True)
class VisualAnalysisResult:
    overall_analysis: str
    mathematical_accuracy: dict[str, Any]
    visual_relevance: dict[str, Any]
    visual_quality: dict[str, Any]
    visual_consistency: dict[str, Any]
    element_layout: dict[str, Any]
    issues: list[VisualIssue]
    raw_response: str = ""

    @property
    def overall_score(self) -> float:
        scores = [data["score"] for _, data in self._dimensions()]
        product = 1.0
        for score in scores:
            product *= score
        # 浮点幂在全 5 分时可能得到 5.000000000000001；收据 schema 的
        # 合法范围仍应严格是 1..5。
        return min(5.0, max(1.0, product ** (1.0 / len(scores))))

    @property
    def has_major_issue(self) -> bool:
        return any(issue.severity == "major" for issue in self.issues)

    def needs_fix(self, threshold: float) -> bool:
        return self.has_major_issue or self.overall_score < threshold

    def feedback(self) -> str:
        """生成给主 Coder 的纯诊断文本，不包含视觉模型生成的代码。"""

        lines = [
            f"视觉评估总分：{self.overall_score:.2f}/5.00",
            f"总体分析：{self.overall_analysis}",
        ]
        for name, data in self._dimensions():
            lines.append(
                f"- {name}: {data['score']}/5 — {data.get('comprehensive_evaluation', '')}"
            )
        for issue in self.issues:
            frames = ", ".join(issue.frame_ids) or "未指定帧"
            lines.append(
                f"- [{issue.severity}/{issue.category}] {frames}: {issue.evidence}；"
                f"建议：{issue.recommendation}"
            )
        return "\n".join(lines)[:20_000]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_analysis": self.overall_analysis,
            "overall_score": round(self.overall_score, 4),
            "has_major_issue": self.has_major_issue,
            "evaluation": {name: data for name, data in self._dimensions()},
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }

    def to_quality_scores(self) -> list[QualityScore]:
        dimensions = (
            (EvalMetric.VISUAL_MATH_ACCURACY, self.mathematical_accuracy),
            (EvalMetric.VISUAL_RELEVANCE, self.visual_relevance),
            (EvalMetric.VISUAL_QUALITY, self.visual_quality),
            (EvalMetric.VISUAL_CONSISTENCY, self.visual_consistency),
            (EvalMetric.ELEMENT_LAYOUT, self.element_layout),
        )
        issue_details = [item.model_dump(mode="json") for item in self.issues]
        return [
            QualityScore(
                metric=metric,
                score=data["score"],
                justification=data.get("comprehensive_evaluation", ""),
                details={**data, "issues": issue_details},
            )
            for metric, data in dimensions
        ]

    def _dimensions(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return (
            ("mathematical_accuracy", self.mathematical_accuracy),
            ("visual_relevance", self.visual_relevance),
            ("visual_quality", self.visual_quality),
            ("visual_consistency", self.visual_consistency),
            ("element_layout", self.element_layout),
        )


class VisualEvaluator:
    """视觉评估器；其 Agent 始终绑定独立视觉 profile。"""

    def __init__(self, model_name: str | None = None):
        self.profile = settings.visual_llm_profile(model_override=model_name)
        self.model_name = self.profile.model
        self._agent: BaseAgent | None = None

    @property
    def agent(self) -> BaseAgent:
        if self._agent is None:
            self._agent = BaseAgent(profile=self.profile)
            self._agent.name = "VisualEvaluator"
        return self._agent

    def check_api_available(self, *, timeout: float | None = None) -> None:
        """用一张 1×1 PNG 验证端点确实接受图片消息。"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image and reply with OK."},
                    {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
                ],
            }
        ]
        self.agent.check_api_available(timeout=timeout, messages=messages)

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

    @staticmethod
    def _normalize_samples(frame_paths: list[Path | FrameSample]) -> list[FrameSample]:
        if not frame_paths:
            raise ValueError("No frame paths provided")
        if len(frame_paths) > MAX_FRAME_COUNT:
            raise ValueError(f"关键帧数量不能超过 {MAX_FRAME_COUNT}")
        samples: list[FrameSample] = []
        for index, item in enumerate(frame_paths, start=1):
            if isinstance(item, FrameSample):
                samples.append(item)
            else:
                samples.append(FrameSample(frame_id=f"F{index:02d}", path=Path(item)))
        ids = [sample.frame_id for sample in samples]
        if len(set(ids)) != len(ids):
            raise ValueError("关键帧 frame_id 必须唯一")
        return samples

    def evaluate_image(
        self,
        image_path: str | Path,
        description: str = "Mathematical animation",
    ) -> VisualAnalysisResult:
        return self.evaluate_video_frames([Path(image_path)], description)

    def evaluate_video_frames(
        self,
        frame_paths: list[Path | FrameSample],
        description: str = "Mathematical animation",
        *,
        scene_context: str = "",
        scope: Literal["scene", "complete video"] = "scene",
    ) -> VisualAnalysisResult:
        samples = self._normalize_samples(frame_paths)
        manifest = [
            {
                "frame_id": sample.frame_id,
                "timestamp_seconds": sample.timestamp_seconds,
                "role": sample.role,
                "filename": sample.path.name,
            }
            for sample in samples
        ]
        prompt = VISUAL_EVAL_PROMPT.format(
            scope=scope,
            description=(description or "Mathematical animation")[:20_000],
            scene_context=(scene_context or "No additional scene context")[:30_000],
            frame_manifest=json.dumps(manifest, ensure_ascii=False),
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for sample in samples:
            if sample.image_sha256 and sha256_file(sample.path) != sample.image_sha256:
                raise ValueError(f"关键帧 {sample.frame_id} 的文件哈希已变化")
            timestamp = (
                f" at {sample.timestamp_seconds:.3f}s"
                if sample.timestamp_seconds is not None
                else ""
            )
            content.append({"type": "text", "text": f"Frame {sample.frame_id}{timestamp} follows."})
            content.append(self._image_content(sample.path))
        messages = [
            {"role": "system", "content": VISUAL_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        structured_call = getattr(self.agent, "call_llm_json", None)
        if callable(structured_call):
            payload = structured_call(
                VISUAL_EVAL_SYSTEM_PROMPT,
                prompt,
                _VisualPayload,
                temperature=self.profile.temperature,
                stream=False,
                messages=messages,
                max_tokens=self.profile.max_tokens,
            )
            result = self._from_payload(payload)
            self._validate_frame_references(result, samples)
            return result
        response = self.agent.call_llm(
            messages=messages,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            json_mode=True,
            stream=False,
        )
        result = self._parse_response(response)
        self._validate_frame_references(result, samples)
        return result

    @staticmethod
    def _validate_frame_references(
        result: VisualAnalysisResult,
        samples: list[FrameSample],
    ) -> None:
        allowed = {sample.frame_id for sample in samples}
        unknown = sorted(
            {
                frame_id
                for issue in result.issues
                for frame_id in issue.frame_ids
                if frame_id not in allowed
            }
        )
        if unknown:
            raise ValueError(f"视觉评估引用了不存在的关键帧: {', '.join(unknown)}")

    @classmethod
    def _from_payload(
        cls, parsed: _VisualPayload, *, raw_response: str = ""
    ) -> VisualAnalysisResult:
        evaluation = parsed.evaluation
        return VisualAnalysisResult(
            overall_analysis=parsed.overall_analysis,
            mathematical_accuracy=evaluation.mathematical_accuracy.model_dump(),
            visual_relevance=evaluation.visual_relevance.model_dump(),
            visual_quality=evaluation.visual_quality.model_dump(),
            visual_consistency=evaluation.visual_consistency.model_dump(),
            element_layout=evaluation.element_layout.model_dump(),
            issues=list(parsed.issues),
            raw_response=raw_response,
        )

    @classmethod
    def _parse_response(cls, response: str) -> VisualAnalysisResult:
        payload = BaseAgent._extract_json(response, expected_type="object")
        parsed = _VisualPayload.model_validate_json(payload)
        return cls._from_payload(parsed, raw_response=response)

    def evaluate_frames(
        self,
        frame_paths: list[Path | FrameSample],
        description: str = "",
        *,
        scene_context: str = "",
        scope: Literal["scene", "complete video"] = "scene",
    ) -> list[QualityScore]:
        result = self.evaluate_video_frames(
            frame_paths,
            description,
            scene_context=scene_context,
            scope=scope,
        )
        return result.to_quality_scores()

    def evaluate(self, image_path: str | Path, description: str = "") -> list[QualityScore]:
        return self.evaluate_frames([Path(image_path)], description)
