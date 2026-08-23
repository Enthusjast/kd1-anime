"""代码、视觉和运行效率的统一评估入口。"""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.eval.code_eval import CodeEvaluator
from kd1_anime.eval.metrics import ComparisonResult, EvalMetric, EvalResult, QualityScore
from kd1_anime.eval.visual_eval import FrameSample, VisualAnalysisResult, VisualEvaluator
from kd1_anime.exceptions import KD1Error
from kd1_anime.rendering import probe_video, sha256_file
from kd1_anime.run_store import RunRepository, atomic_write_json, restore_run_path


class EvaluationError(KD1Error):
    pass


MAX_VISUAL_FRAMES = 8
MAX_VISUAL_FRAME_BYTES = 2 * 1024 * 1024


def _validate_run_id_for_base_dir(run_id: str) -> None:
    """即使调用方显式提供 base_dir，也拒绝路径遍历和绝对路径。"""

    candidate = Path(run_id)
    if not run_id or candidate.is_absolute() or candidate.name != run_id or ".." in candidate.parts:
        raise ValueError(f"run_id 包含不安全路径: {run_id!r}")


class Evaluator:
    def __init__(
        self,
        enable_visual_eval: bool = True,
        visual_eval_model: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.code_evaluator = CodeEvaluator()
        self.visual_evaluator = (
            VisualEvaluator(visual_eval_model) if enable_visual_eval else None
        )
        self.output_dir = output_dir or (
            resolve_runtime_path(settings.WORKSPACE_DIR) / "eval_results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.chmod(0o700)

    def evaluate_code(self, code: str) -> EvalResult:
        result = EvalResult(run_id=f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        for score in self.code_evaluator.evaluate(code):
            result.add_score(score)
        result.summary = self._generate_summary(result)
        return result

    def evaluate_visual(self, image_path: str | Path, description: str = "") -> EvalResult:
        if self.visual_evaluator is None:
            raise EvaluationError("Visual evaluation is disabled")
        result = EvalResult(run_id=f"visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        for score in self.visual_evaluator.evaluate(image_path, description):
            result.add_score(score)
        result.summary = self._generate_summary(result)
        return result

    @staticmethod
    def extract_video_samples(
        video_path: Path,
        output_dir: Path,
        *,
        frame_count: int = 6,
    ) -> list[FrameSample]:
        if not 1 <= frame_count <= MAX_VISUAL_FRAMES:
            raise ValueError(f"frame_count 必须在 1..{MAX_VISUAL_FRAMES} 之间")
        metadata = probe_video(video_path)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise EvaluationError("未找到 ffmpeg，无法抽取视觉评估关键帧")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.chmod(0o700)
        samples: list[FrameSample] = []
        if frame_count == 1:
            positions = [0.5]
        else:
            positions = [0.05 + index * 0.9 / (frame_count - 1) for index in range(frame_count)]
        for index in range(frame_count):
            timestamp = metadata.duration_seconds * positions[index]
            output = output_dir / f"frame_{index + 1:02d}.jpg"
            try:
                result = subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(video_path),
                        "-vf",
                        "scale=1024:1024:force_original_aspect_ratio=decrease",
                        "-frames:v",
                        "1",
                        "-c:v",
                        "mjpeg",
                        "-q:v",
                        "5",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise EvaluationError(f"关键帧抽取超时 ({index + 1}/{frame_count})") from exc
            if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                raise EvaluationError(
                    f"关键帧抽取失败 ({index + 1}/{frame_count}): {result.stderr[-500:]}"
                )
            if output.stat().st_size > MAX_VISUAL_FRAME_BYTES:
                raise EvaluationError(
                    f"关键帧过大 ({index + 1}/{frame_count}): "
                    f"{output.stat().st_size} bytes > {MAX_VISUAL_FRAME_BYTES}"
                )
            output.chmod(0o600)
            samples.append(
                FrameSample(
                    frame_id=f"F{index + 1:02d}",
                    path=output,
                    timestamp_seconds=timestamp,
                    image_sha256=sha256_file(output),
                )
            )
        return samples

    @staticmethod
    def extract_video_frames(
        video_path: Path,
        output_dir: Path,
        *,
        frame_count: int = 6,
    ) -> list[Path]:
        """兼容旧 API；新场景评估使用带时间戳的 extract_video_samples。"""

        return [
            sample.path
            for sample in Evaluator.extract_video_samples(
                video_path,
                output_dir,
                frame_count=frame_count,
            )
        ]

    def evaluate_scene_video(
        self,
        video_path: Path,
        *,
        description: str,
        scene_context: str,
        output_dir: Path,
        frame_count: int | None = None,
    ) -> tuple[VisualAnalysisResult, list[FrameSample]]:
        """评估一个经过上游哈希校验的精确场景视频。"""

        if self.visual_evaluator is None:
            raise EvaluationError("Visual evaluation is disabled")
        samples = self.extract_video_samples(
            video_path,
            output_dir,
            frame_count=frame_count or settings.VISUAL_EVAL_FRAME_COUNT,
        )
        result = self.visual_evaluator.evaluate_video_frames(
            samples,
            description,
            scene_context=scene_context,
            scope="scene",
        )
        return result, samples

    def evaluate_run_scene(
        self,
        run_id: str,
        scene_id: int,
        *,
        description: str = "",
        frame_count: int | None = None,
    ) -> EvalResult:
        """按 manifest 的精确 SceneArtifact 评估单个已渲染场景。"""

        if self.visual_evaluator is None:
            raise EvaluationError("Visual evaluation is disabled")
        repository = RunRepository(settings.WORKSPACE_DIR)
        manifest = repository.load(run_id)
        run_dir = repository.run_root(run_id)
        scene = manifest.scenes.get(scene_id)
        if scene is None:
            raise EvaluationError(f"Run {run_id} 不包含 Scene {scene_id}")
        artifact = scene.artifact
        if not scene.rendered or artifact is None or not artifact.verified:
            raise EvaluationError(f"Scene {scene_id} 没有可验证的渲染产物")
        if (
            artifact.scene_id != scene_id
            or artifact.scene_class_name != scene.class_name
            or artifact.code_sha256 != scene.code_sha256
            or artifact.render_profile_sha256 != manifest.render_profile.digest()
        ):
            raise EvaluationError(f"Scene {scene_id} 的渲染产物身份与运行清单不一致")
        source_root = (
            run_dir
            if artifact.source_run_id == run_id
            else repository.run_root(artifact.source_run_id)
        )
        video = restore_run_path(source_root, artifact.video_path)
        if not video.is_file() or video.stat().st_size <= 0:
            raise EvaluationError(f"Scene {scene_id} 的渲染视频不存在")
        if sha256_file(video) != artifact.video_sha256:
            raise EvaluationError(f"Scene {scene_id} 的渲染视频哈希不匹配")

        samples_dir = (
            run_dir
            / "eval_frames"
            / f"manual_scene_{scene_id}"
            / artifact.video_sha256[:12]
        )
        analysis, samples = self.evaluate_scene_video(
            video,
            description=description or manifest.user_prompt,
            scene_context=json.dumps(scene.plan.model_dump(mode="json"), ensure_ascii=False),
            output_dir=samples_dir,
            frame_count=frame_count,
        )
        result = EvalResult(
            run_id=f"{run_id}:scene:{scene_id}",
            metadata={
                "run_id": run_id,
                "scene_id": scene_id,
                "video_sha256": artifact.video_sha256,
                "frames": [
                    {
                        "frame_id": sample.frame_id,
                        "timestamp_seconds": sample.timestamp_seconds,
                        "path": sample.path.relative_to(run_dir).as_posix(),
                        "sha256": sample.image_sha256,
                    }
                    for sample in samples
                ],
                "overall_analysis": analysis.overall_analysis,
            },
        )
        for score in analysis.to_quality_scores():
            result.add_score(score)
        result.summary = self._generate_summary(result)
        return result

    def evaluate_run(
        self,
        run_id: str,
        run_dir: Path | None = None,
        description: str = "",
        enable_visual: bool = True,
    ) -> EvalResult:
        repository = RunRepository(settings.WORKSPACE_DIR)
        if run_dir is None:
            manifest = repository.load(run_id)
            run_dir = repository.run_root(run_id)
        else:
            _validate_run_id_for_base_dir(run_id)
            run_dir = run_dir.resolve()
            if not run_dir.is_dir():
                raise EvaluationError(f"Run directory not found: {run_dir}")
            manifest = None
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = (
                        repository.load(run_id) if run_dir == repository.run_root(run_id) else None
                    )
                except (FileNotFoundError, ValueError):
                    manifest = None

        result = EvalResult(
            run_id=run_id,
            metadata={"run_dir": str(run_dir), "description": description},
        )
        code_files = sorted((run_dir / "scenes").glob("scene_*.py"))
        if not code_files:
            code_files = sorted(run_dir.glob("scene_*.py"))
        for code_file in code_files:
            try:
                for score in self.code_evaluator.evaluate(code_file.read_text(encoding="utf-8")):
                    score.details["file"] = str(code_file)
                    result.add_score(score)
            except (OSError, UnicodeError, ValueError) as exc:
                result.add_error(f"code:{code_file.name}", str(exc))

        if enable_visual and self.visual_evaluator is not None:
            final_video = None
            manifest_video_rejected = False
            if manifest and manifest.final_video:
                candidate = Path(manifest.final_video).expanduser().resolve()
                run_root = run_dir.resolve()
                allowed_external = Path(manifest.output_path).expanduser().resolve()
                try:
                    candidate.relative_to(run_root)
                    is_allowed = True
                except ValueError:
                    is_allowed = candidate == allowed_external
                if not is_allowed:
                    manifest_video_rejected = True
                    result.add_error("visual", "清单中的最终视频路径不在允许范围内")
                elif not candidate.is_file():
                    manifest_video_rejected = True
                    result.add_error("visual", f"清单中的最终视频不存在: {candidate}")
                else:
                    try:
                        hash_matches = not manifest.final_video_sha256 or (
                            sha256_file(candidate) == manifest.final_video_sha256
                        )
                    except OSError as exc:
                        hash_matches = False
                        manifest_video_rejected = True
                        result.add_error("visual", f"读取最终视频失败，拒绝视觉评估: {exc}")
                    if not hash_matches and "visual" not in result.errors:
                        manifest_video_rejected = True
                        result.add_error("visual", "清单中的最终视频哈希不匹配，拒绝视觉评估")
                    if hash_matches:
                        final_video = candidate
            if final_video is None and not manifest_video_rejected:
                candidate = run_dir / "output_final.mp4"
                if candidate.is_file():
                    final_video = candidate
            if final_video:
                try:
                    frames = self.extract_video_frames(final_video, run_dir / "eval_frames")
                    for score in self.visual_evaluator.evaluate_frames(frames, description):
                        result.add_score(score)
                except Exception as exc:
                    result.add_error("visual", str(exc))
            elif final_video is None and not result.errors.get("visual"):
                result.add_error("visual", "未找到最终视频，无法进行视觉评估")

        for score in self._evaluate_efficiency(run_dir, manifest):
            result.add_score(score)
        result.summary = self._generate_summary(result)
        return result

    @staticmethod
    def _score_render_time(seconds: float) -> int:
        if seconds < 30:
            return 5
        if seconds < 60:
            return 4
        if seconds < 120:
            return 3
        if seconds < 300:
            return 2
        return 1

    def _evaluate_efficiency(self, run_dir: Path, manifest=None) -> list[QualityScore]:
        raw: dict[str, Any] = {}
        if manifest is None:
            try:
                raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
        scenes = manifest.scenes if manifest is not None else raw.get("scenes", {})
        if scenes:
            values = list(scenes.values())
            rendered = sum(
                1
                for scene in values
                if (scene.rendered if manifest is not None else scene.get("rendered", False))
            )
            total = len(values)
            retries = sum(
                scene.fix_attempts if manifest is not None else int(scene.get("fix_attempts", 0))
                for scene in values
            )
            elapsed_values = []
            for scene in values:
                job = scene.slurm_job if manifest is not None else scene.get("slurm_job")
                if job:
                    elapsed = (
                        job.elapsed_seconds if manifest is not None else job.get("elapsed_seconds")
                    )
                    if elapsed:
                        elapsed_values.append(float(elapsed))
        else:
            # 兼容旧的独立评估 fixture。
            total = int(raw.get("total_scenes", 0))
            rendered = int(raw.get("successful_scenes", 0))
            retries = int(raw.get("retry_count", 0))
            elapsed_values = (
                [float(raw["render_time_seconds"])] if raw.get("render_time_seconds") else []
            )

        scores: list[QualityScore] = []
        if elapsed_values:
            render_time = sum(elapsed_values)
            scores.append(
                QualityScore(
                    EvalMetric.RENDER_TIME,
                    self._score_render_time(render_time),
                    f"Render time: {render_time:.1f} seconds",
                    {"render_time_seconds": render_time},
                )
            )
        if total:
            success_rate = rendered / total
            rate_score = (
                5
                if success_rate >= 0.95
                else 4
                if success_rate >= 0.8
                else 3
                if success_rate >= 0.6
                else 2
                if success_rate >= 0.4
                else 1
            )
            scores.append(
                QualityScore(
                    EvalMetric.SUCCESS_RATE,
                    rate_score,
                    f"Success rate: {success_rate:.1%} ({rendered}/{total})",
                    {
                        "total_scenes": total,
                        "successful_scenes": rendered,
                        "success_rate": success_rate,
                    },
                )
            )
        if total:
            retry_score = (
                5
                if retries <= 1
                else 4
                if retries <= 3
                else 3
                if retries <= 5
                else 2
                if retries <= 10
                else 1
            )
            scores.append(
                QualityScore(
                    EvalMetric.RETRY_COUNT,
                    retry_score,
                    f"Total retries: {retries}",
                    {"retry_count": retries},
                )
            )
        return scores

    @staticmethod
    def _generate_summary(result: EvalResult) -> str:
        if result.overall_score is None:
            return "Overall score: unknown"
        parts = [f"Overall score: {result.overall_score:.2f}/5.00"]
        for category in ("code", "visual", "render", "success", "retry"):
            scores = result.get_scores_by_category(category)
            if scores:
                parts.append(
                    f"{category.capitalize()}: "
                    f"{sum(score.score for score in scores) / len(scores):.1f}/5.0"
                )
        if result.errors:
            parts.append("Unknown: " + ", ".join(sorted(result.errors)))
        return " | ".join(parts)

    def evaluate_batch(
        self,
        run_ids: list[str],
        base_dir: Path | None = None,
        description: str = "",
        max_workers: int = 4,
    ) -> list[EvalResult]:
        if max_workers < 1:
            raise ValueError("max_workers 必须大于 0")
        if base_dir is None:
            repository = RunRepository(settings.WORKSPACE_DIR)
            # 默认 workspace 是受信任的运行存储；不要让 API 调用者通过
            # run_id=../... 把批量评估读到 workspace 之外。
            for run_id in run_ids:
                repository.run_root(run_id)
            base_dir = repository.runs_root
        else:
            base_dir = base_dir.resolve()
            for run_id in run_ids:
                _validate_run_id_for_base_dir(run_id)
                candidate = (base_dir / run_id).resolve()
                try:
                    candidate.relative_to(base_dir)
                except ValueError as exc:
                    raise ValueError(f"run_id 越出 base_dir: {run_id!r}") from exc
        results: dict[int, EvalResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_run,
                    run_id,
                    base_dir / run_id,
                    description,
                    False,
                ): (index, run_id)
                for index, run_id in enumerate(run_ids)
            }
            for future in as_completed(futures):
                index, run_id = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    unknown = EvalResult(run_id=run_id)
                    unknown.add_error("evaluation", str(exc))
                    unknown.summary = self._generate_summary(unknown)
                    results[index] = unknown
        return [results[index] for index in range(len(run_ids))]

    def compare_runs(
        self,
        baseline_run_id: str,
        current_run_id: str,
        base_dir: Path | None = None,
    ) -> ComparisonResult:
        if base_dir is None:
            repository = RunRepository(settings.WORKSPACE_DIR)
            repository.run_root(baseline_run_id)
            repository.run_root(current_run_id)
            base_dir = repository.runs_root
        else:
            base_dir = base_dir.resolve()
            for run_id in (baseline_run_id, current_run_id):
                _validate_run_id_for_base_dir(run_id)
                candidate = (base_dir / run_id).resolve()
                try:
                    candidate.relative_to(base_dir)
                except ValueError as exc:
                    raise ValueError(f"run_id 越出 base_dir: {run_id!r}") from exc
        baseline = self.evaluate_run(baseline_run_id, base_dir / baseline_run_id)
        current = self.evaluate_run(current_run_id, base_dir / current_run_id)
        improvements: list[str] = []
        regressions: list[str] = []
        for metric in EvalMetric:
            before = baseline.get_metric_average(metric)
            after = current.get_metric_average(metric)
            if before is not None and after is not None and before != after:
                target = improvements if after > before else regressions
                target.append(f"{metric.value}: {before:.2f} → {after:.2f}")
        return ComparisonResult(
            baseline_run_id,
            current_run_id,
            baseline,
            current,
            improvements,
            regressions,
        )

    def generate_report(
        self,
        results: list[EvalResult],
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or (
            self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        numeric = [result.overall_score for result in results if result.overall_score is not None]
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(results),
            "average_score": sum(numeric) / len(numeric) if numeric else None,
            "results": [result.to_dict() for result in results],
            "summary": self._generate_batch_summary(results),
        }
        atomic_write_json(output_path, report)
        return output_path

    @staticmethod
    def _generate_batch_summary(results: list[EvalResult]) -> dict[str, Any]:
        rated = [result for result in results if result.overall_score is not None]
        if not rated:
            return {"rated_runs": 0, "unknown_runs": len(results)}
        best = max(rated, key=lambda item: item.overall_score or 0)
        worst = min(rated, key=lambda item: item.overall_score or 0)
        scores = [item.overall_score or 0 for item in rated]
        return {
            "rated_runs": len(rated),
            "unknown_runs": len(results) - len(rated),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "best_run": best.run_id,
            "worst_run": worst.run_id,
        }
