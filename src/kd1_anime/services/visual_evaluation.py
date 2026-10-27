"""Visual evaluation over explicit video and frame identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kd1_anime.rendering import sha256_file

if TYPE_CHECKING:
    from kd1_anime.eval.visual_eval import FrameSample, VisualAnalysisResult


class VisualIdentityError(ValueError):
    """An evidence identity mismatch must never become an unknown provider result."""


def verify_samples(samples) -> None:
    for sample in samples:
        if (
            not sample.image_sha256
            or sample.path.is_symlink()
            or not sample.path.is_file()
            or sha256_file(sample.path) != sample.image_sha256
        ):
            raise VisualIdentityError("视觉评估帧摘要不匹配")


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
        if inputs.video.is_symlink() or sha256_file(inputs.video) != inputs.video_sha256:
            raise VisualIdentityError("视觉输入视频摘要不匹配")
        result, samples = evaluator.evaluate_scene_video(
            inputs.video,
            description=inputs.description,
            scene_context=inputs.scene_context,
            output_dir=output_dir,
            frame_count=frame_count,
        )
        if inputs.video.is_symlink() or sha256_file(inputs.video) != inputs.video_sha256:
            raise VisualIdentityError("视觉评估期间视频已变化")
        verify_samples(samples)
        return SceneVisualResult(inputs, result, tuple(samples))

    @staticmethod
    def evaluate_frames(evaluator, samples, description: str, *, scene_context: str, scope: str):
        if evaluator is None:
            raise RuntimeError("视觉评估器未初始化")
        verify_samples(samples)
        result = evaluator.evaluate_video_frames(
            samples, description, scene_context=scene_context, scope=scope
        )
        verify_samples(samples)
        return result


__all__ = [
    "SceneVisualInput",
    "SceneVisualResult",
    "VisualEvaluationService",
    "VisualIdentityError",
    "verify_samples",
]
