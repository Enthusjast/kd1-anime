"""Exact current-job merging and private immutable final-video identities."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from kd1_anime.agent_runtime.contracts import ObjectRef, sanitize_record
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.eval.metrics import EvalResult
from kd1_anime.media.merger import VideoMerger
from kd1_anime.pipeline.contracts import FinalizationInputs, FinalizationOutput
from kd1_anime.pipeline.render import validate_artifact
from kd1_anime.pipeline.visual import (
    _PROPAGATE,
    VisualPipeline,
    _preserve_frames,
    _read_report,
    _secrets,
    _source,
    frame_bindings,
    scene_binding,
    store_report,
    verify_scene_receipt,
    visual_profile,
)
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import restore_run_path, sha256_text
from kd1_anime.services.visual_evaluation import verify_samples


def _immutable_video(root: Path, source: Path) -> ObjectRef:
    """Stream media to a content-addressed private file without a mutable hardlink."""
    digest = sha256_file(source)
    relative = f"artifacts/videos/{digest}.mp4"
    target = restore_run_path(root, relative)
    directory = target.parent
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for parent in reversed(missing):
        parent.mkdir(mode=0o700, exist_ok=True)
        parent.chmod(0o700)
    if target.exists():
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError("Immutable final video identity mismatch")
        return ObjectRef(path=relative, sha256=digest, kind="video")
    descriptor, name = tempfile.mkstemp(prefix=".video-", dir=directory)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as original:
            shutil.copyfileobj(original, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != digest:
            raise ValueError("Final video changed while preserving its immutable identity")
        try:
            os.link(temporary, target)
        except FileExistsError:
            restore_run_path(root, relative)
            if not target.is_file() or sha256_file(target) != digest:
                raise ValueError("Immutable final video identity mismatch") from None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return ObjectRef(path=relative, sha256=digest, kind="video")


class FinalizationPipeline:
    def __init__(
        self,
        merger,
        *,
        evaluator_factory=None,
        fixer_factory=None,
        request_context=None,
        control=None,
    ):
        self.merger = merger
        from kd1_anime.eval import Evaluator

        self.visual = VisualPipeline(
            evaluator_factory=evaluator_factory or Evaluator,
            fixer_factory=fixer_factory,
            request_context=request_context,
            control=control,
        )

    @staticmethod
    def _merge_inputs(inputs: FinalizationInputs) -> tuple[list[Path], Path]:
        if not inputs.scenes:
            raise ValueError("Finalization requires verified scene artifacts")
        scene_ids = [scene.plan.scene_id for scene in inputs.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Finalization contains duplicate scene identities")
        settings = inputs.config.resolve()
        incomplete = [scene.plan.scene_id for scene in inputs.scenes if not scene.rendered]
        if incomplete and not (inputs.allow_partial or settings.ALLOW_PARTIAL_OUTPUT):
            raise ValueError(f"Partial output is disabled; incomplete scenes: {incomplete}")
        rendered = sorted(
            (scene for scene in inputs.scenes if scene.rendered),
            key=lambda item: item.plan.scene_id,
        )
        if not rendered:
            raise ValueError("Finalization requires at least one verified scene artifact")
        videos = []
        for scene in rendered:
            _source(inputs, scene)
            videos.append(
                validate_artifact(
                    inputs.root,
                    inputs.run_id,
                    scene,
                    inputs.render_profile,
                    provenance=inputs.reuse_receipts.get(scene.plan.scene_id),
                )
            )
            if visual_profile(inputs).enabled:
                if scene.visual_status not in {"passed", "warning", "unknown"}:
                    raise ValueError("Current scene has not completed its visual gate")
                verify_scene_receipt(inputs, scene)
        requested = inputs.output_path.expanduser().absolute()
        if requested.is_symlink() or requested.resolve() in videos:
            raise ValueError("Final output must not alias an input artifact")
        return videos, requested

    def merge(self, inputs: FinalizationInputs, context: ToolContext) -> FinalizationOutput:
        context.checkpoint()
        videos, requested = self._merge_inputs(inputs)
        with inputs.config.activate():
            # VideoMerger owns the output lock, ffprobe verification and atomic replace.
            merged = Path(
                self.merger.merge(
                    videos,
                    requested,
                    render_profile=inputs.render_profile,
                    merge_profile=inputs.merge_profile,
                    replace_existing=requested.resolve().is_relative_to(inputs.root.resolve())
                    or bool(
                        inputs.existing_output_sha256
                        and requested.is_file()
                        and sha256_file(requested) == inputs.existing_output_sha256
                    ),
                )
            )
        if (
            merged.is_symlink()
            or merged.resolve() != requested.resolve()
            or not merged.is_file()
            or merged.stat().st_size <= 0
        ):
            raise ValueError("Merger returned an invalid final output identity")
        reference = _immutable_video(inputs.root, merged)
        return FinalizationOutput(video=reference, output_path=str(requested.resolve()))

    def reconcile_merge(
        self, inputs: FinalizationInputs, context: ToolContext
    ) -> FinalizationOutput:
        """Recompute an unknown merge result, then publish only byte-authorized output.

        A valid-looking public video is not evidence of this merge. Only the exact
        candidate hash or the caller's previously authorized same-run hash permits
        adoption/replacement. No media discovery or generic retry is performed.
        """
        context.checkpoint()
        videos, requested = self._merge_inputs(inputs)
        with tempfile.TemporaryDirectory(prefix=".merge-recovery-", dir=inputs.root) as name:
            candidate = Path(name) / "candidate.mp4"
            with inputs.config.activate():
                merged = Path(
                    self.merger.merge(
                        videos,
                        candidate,
                        render_profile=inputs.render_profile,
                        merge_profile=inputs.merge_profile,
                        replace_existing=False,
                    )
                )
            if (
                merged.is_symlink()
                or merged.resolve() != candidate.resolve()
                or not merged.is_file()
                or merged.stat().st_size <= 0
            ):
                raise ValueError("Merger returned an invalid recovery candidate identity")
            # Reject source/artifact mutations during the recomputation as well.
            self._merge_inputs(inputs)
            reference = _immutable_video(inputs.root, merged)
            private = restore_run_path(inputs.root, reference.path)
            context.checkpoint()
            requested.parent.mkdir(parents=True, exist_ok=True)
            # Use precisely the lock used by normal VideoMerger publication.
            with VideoMerger._output_lock(requested.resolve()):
                self._publish_reconciled(inputs, private, requested, reference.sha256)
            return FinalizationOutput(video=reference, output_path=str(requested.resolve()))

    @staticmethod
    def _output_digest(requested: Path) -> str | None:
        if requested.is_symlink():
            raise ValueError("Final output must not be a symbolic link")
        if not requested.exists():
            return None
        if not requested.is_file():
            raise ValueError("Final output must be a regular file")
        return sha256_file(requested)

    @classmethod
    def _publish_reconciled(cls, inputs, private, requested, digest):
        """Publish under VideoMerger's lock; an absent target uses no-clobber link."""
        existing = cls._output_digest(requested)
        if existing != digest:
            if existing is not None and existing != inputs.existing_output_sha256:
                raise ValueError("Existing final output differs from the verified merge candidate")
            descriptor, name = tempfile.mkstemp(prefix=".merge-recovery-", dir=requested.parent)
            temporary = Path(name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as output, private.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256_file(temporary) != digest:
                    raise ValueError("Immutable recovery candidate hash mismatch")
                if existing is None:
                    try:
                        # Atomic even against a writer that does not honor our lock.
                        os.link(temporary, requested)
                    except FileExistsError:
                        if cls._output_digest(requested) != digest:
                            raise ValueError("Final output appeared with unrelated bytes") from None
                else:
                    # Authorization is hash-bound, not merely path- or run-bound.
                    current = cls._output_digest(requested)
                    if current != digest:
                        if current != inputs.existing_output_sha256:
                            raise ValueError("Previously authorized final output changed")
                        os.replace(temporary, requested)
            finally:
                temporary.unlink(missing_ok=True)
        if cls._output_digest(requested) != digest:
            raise ValueError("Final output changed during merge reconciliation")
        requested.chmod(0o600)
        with requested.open("rb") as output:
            os.fsync(output.fileno())
        directory_fd = os.open(requested.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def evaluate(
        self, inputs: FinalizationInputs, video: FinalizationOutput, context: ToolContext
    ) -> FinalizationOutput:
        context.checkpoint()
        self._validate_video(inputs, video)
        settings = inputs.config.resolve()
        changes = {}
        if visual_profile(inputs).enabled and video.visual_report is None:
            reference, status = self._final_visual(inputs, video, context)
            changes.update(visual_report=reference, visual_status=status)
        if settings.ENABLE_AUTO_EVAL and video.evaluation_report is None:
            reference, scenes, eval_round = self._automatic_evaluation(inputs, video, context)
            changes.update(
                evaluation_report=reference, scenes_to_improve=scenes, eval_round=eval_round
            )
        self._validate_video(inputs, video)
        return video.model_copy(update=changes)

    @staticmethod
    def _validate_video(inputs, video):
        if video.video.kind != "video":
            raise ValueError("Final output reference must identify a video")
        private = restore_run_path(inputs.root, video.video.path)
        requested = Path(video.output_path)
        if (
            requested.is_symlink()
            or requested.resolve() != inputs.output_path.expanduser().resolve()
            or not private.is_file()
            or not requested.is_file()
            or private.stat().st_size <= 0
            or sha256_file(private) != video.video.sha256
            or sha256_file(requested) != video.video.sha256
        ):
            raise ValueError("Final video hash or output identity mismatch")
        for scene in inputs.scenes:
            if scene.rendered:
                _source(inputs, scene)
                validate_artifact(
                    inputs.root,
                    inputs.run_id,
                    scene,
                    inputs.render_profile,
                    provenance=inputs.reuse_receipts.get(scene.plan.scene_id),
                )
        for reference, scope in (
            (video.visual_report, "complete_video"),
            (video.evaluation_report, "run_evaluation"),
        ):
            if reference is None:
                continue
            report = _read_report(inputs, reference)
            scenes = (
                [s for s in inputs.scenes if s.rendered]
                if scope == "complete_video"
                else inputs.scenes
            )
            if (
                report.get("scope") != scope
                or report.get("video_sha256") != video.video.sha256
                or report.get("scene_bindings") != [scene_binding(s) for s in scenes]
            ):
                raise ValueError("Final report does not bind the current video and scenes")
            if scope == "complete_video" and (
                report.get("visual_profile_sha256") != visual_profile(inputs).digest()
                or report.get("state_ledger_digest") != inputs.state_ledger.digest()
            ):
                raise ValueError("Final visual report context identity mismatch")
            if scope == "run_evaluation" and report.get("config_sha256") != sha256_text(
                inputs.config.values_json
            ):
                raise ValueError("Final evaluation configuration identity mismatch")
            for frame in report.get("frames", []):
                path = restore_run_path(inputs.root, frame["path"])
                if (
                    not frame.get("sha256")
                    or not path.is_file()
                    or sha256_file(path) != frame["sha256"]
                ):
                    raise ValueError("Final report frame hash mismatch")

    def _final_visual(self, inputs, video, context):
        profile = visual_profile(inputs)
        samples, result, error = [], None, ""
        rendered = [
            (
                scene.plan.scene_id,
                validate_artifact(
                    inputs.root,
                    inputs.run_id,
                    scene,
                    inputs.render_profile,
                    provenance=inputs.reuse_receipts.get(scene.plan.scene_id),
                ),
            )
            for scene in sorted(inputs.scenes, key=lambda item: item.plan.scene_id)
            if scene.rendered
        ]
        with inputs.config.activate():
            try:
                evaluator = self.visual.evaluator(inputs, context)
                boundary_count = min(3, profile.frame_count // 2) if len(rendered) >= 2 else 0
                regular_count = profile.frame_count - 2 * boundary_count
                private = restore_run_path(inputs.root, video.video.path)
                regular = (
                    evaluator.extract_video_samples(
                        private,
                        inputs.root / "eval_frames" / "final" / video.video.sha256,
                        frame_count=regular_count,
                    )
                    if regular_count
                    else []
                )
                boundaries = (
                    evaluator.extract_boundary_samples(
                        rendered,
                        inputs.root / "eval_frames" / "final_boundaries" / video.video.sha256,
                        max_boundaries=boundary_count,
                    )
                    if boundary_count
                    else []
                )
                samples = [
                    sample.model_copy(update={"frame_id": f"F{index:02d}"})
                    for index, sample in enumerate([*regular, *boundaries], start=1)
                ]
                samples = _preserve_frames(inputs, samples)
                if not samples:
                    raise RuntimeError("没有可用于成片视觉评估的关键帧")
                result = self.visual.service.evaluate_frames(
                    evaluator.visual_evaluator,
                    samples,
                    self.visual._description(inputs),
                    scene_context=json.dumps(
                        {
                            "lesson_spec": inputs.lesson_spec.model_dump(mode="json")
                            if inputs.lesson_spec
                            else None,
                            "teaching_graph": inputs.teaching_graph.model_dump(mode="json")
                            if inputs.teaching_graph
                            else None,
                            "scene_plans": [s.plan.model_dump(mode="json") for s in inputs.scenes],
                        },
                        ensure_ascii=False,
                    )[:30_000],
                    scope="complete video",
                )
            except _PROPAGATE:
                raise
            except Exception as exc:
                error = sanitize_record(str(exc) or type(exc).__name__, _secrets())
            self.visual._checkpoint(context)
            self._validate_video(inputs, video)
            verify_samples(samples)
            reference = store_report(
                inputs,
                {
                    "schema_version": 1,
                    "scope": "complete_video",
                    "model": profile.model,
                    "visual_profile_sha256": profile.digest(),
                    "video_sha256": video.video.sha256,
                    "scene_bindings": [scene_binding(s) for s in inputs.scenes if s.rendered],
                    "state_ledger_digest": inputs.state_ledger.digest(),
                    "frames": frame_bindings(inputs, samples),
                    "result": result.to_dict() if result is not None else None,
                    "error": error,
                },
            )
        status = (
            "unknown"
            if result is None
            else ("rejected" if result.needs_fix(profile.threshold) else "accepted")
        )
        # Final-video evidence cannot reliably attribute a rewrite to one scene.
        return reference, status

    def _automatic_evaluation(self, inputs, video, context):
        settings = inputs.config.resolve()
        result = EvalResult(run_id=inputs.run_id)
        scene_results, error = {}, ""
        with inputs.config.activate():
            try:
                evaluator = self.visual.evaluator(inputs, context, visual=False)
                for scene in inputs.scenes:
                    if not scene.code_file:
                        continue
                    self.visual._checkpoint(context)
                    evaluated = evaluator.evaluate_code(
                        _source(inputs, scene), renderer=inputs.render_profile.renderer
                    )
                    scene_results[scene.plan.scene_id] = evaluated
                    for score in evaluated.scores:
                        result.add_score(score)
                # Supply only the immutable stage snapshot: never reload a manifest
                # or discover videos from shared directories during evaluation.
                snapshot = SimpleNamespace(scenes={s.plan.scene_id: s for s in inputs.scenes})
                for score in evaluator._evaluate_efficiency(inputs.root, manifest=snapshot):
                    result.add_score(score)
            except _PROPAGATE:
                raise
            except Exception as exc:
                error = sanitize_record(str(exc) or type(exc).__name__, _secrets())
                result.add_error("evaluation", error)
            self.visual._checkpoint(context)
            self._validate_video(inputs, video)
            low = tuple(
                sorted(
                    scene_id
                    for scene_id, evaluated in scene_results.items()
                    if evaluated.overall_score is not None
                    and evaluated.overall_score < settings.EVAL_THRESHOLD
                )
            )
            improve = (
                not error
                and inputs.auto_fix
                and inputs.eval_round < settings.MAX_EVAL_ROUNDS
                and result.overall_score is not None
                and result.overall_score < settings.EVAL_THRESHOLD
                and bool(low)
            )
            targets = low if improve else ()
            next_round = inputs.eval_round + 1 if improve else inputs.eval_round
            reference = store_report(
                inputs,
                {
                    "schema_version": 1,
                    "scope": "run_evaluation",
                    "video_sha256": video.video.sha256,
                    "scene_bindings": [scene_binding(s) for s in inputs.scenes],
                    "config_sha256": sha256_text(inputs.config.values_json),
                    "eval_round": next_round,
                    "scenes_to_improve": targets,
                    "result": result.to_dict(),
                    "error": error,
                },
            )
        return reference, targets, next_round
