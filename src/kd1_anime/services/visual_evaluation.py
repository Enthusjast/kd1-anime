"""Visual evaluation over explicit video and frame identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kd1_anime.rendering import sha256_file

if TYPE_CHECKING:
    from kd1_anime.eval.visual_eval import FrameSample, VisualAnalysisResult


@dataclass(frozen=True, slots=True)
class SceneVisualInput:
    video: Path
    video_sha256: str
    code_sha256: str
    inherited_sha256: str
    description: str
    scene_context: str


@dataclass(frozen=True, slots=True)
class SceneVisualResult:
    inputs: SceneVisualInput
    analysis: VisualAnalysisResult
    samples: tuple[FrameSample, ...]


class VisualEvaluationService:
    @staticmethod
    def evaluate_scene(
        evaluator, inputs: SceneVisualInput, *, output_dir: Path, frame_count: int
    ) -> SceneVisualResult:
        if sha256_file(inputs.video) != inputs.video_sha256:
            raise ValueError("视觉输入视频摘要不匹配")
        result, samples = evaluator.evaluate_scene_video(
            inputs.video,
            description=inputs.description,
            scene_context=inputs.scene_context,
            output_dir=output_dir,
            frame_count=frame_count,
        )
        if sha256_file(inputs.video) != inputs.video_sha256:
            raise ValueError("视觉评估期间视频已变化")
        for sample in samples:
            if sample.image_sha256 and sha256_file(sample.path) != sample.image_sha256:
                raise ValueError("视觉评估帧摘要不匹配")
        return SceneVisualResult(inputs, result, tuple(samples))

    @staticmethod
    def evaluate_frames(evaluator, samples, description: str, *, scene_context: str, scope: str):
        if evaluator is None:
            raise RuntimeError("视觉评估器未初始化")
        identities = [
            (sample.path, sample.image_sha256)
            for sample in samples
            if getattr(sample, "image_sha256", "")
        ]
        for path, expected in identities:
            if sha256_file(path) != expected:
                raise ValueError("视觉评估帧摘要不匹配")
        result = evaluator.evaluate_video_frames(
            samples, description, scene_context=scene_context, scope=scope
        )
        for path, expected in identities:
            if sha256_file(path) != expected:
                raise ValueError("视觉评估期间帧已变化")
        return result


__all__ = ["SceneVisualInput", "SceneVisualResult", "VisualEvaluationService"]
