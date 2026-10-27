"""Exact current-job merging and private immutable final-video identities."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from kd1_anime.agent_runtime.contracts import ObjectRef
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.pipeline.contracts import FinalizationInputs, FinalizationOutput
from kd1_anime.pipeline.render import validate_artifact
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import restore_run_path


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
    def __init__(self, merger):
        self.merger = merger

    def merge(self, inputs: FinalizationInputs, context: ToolContext) -> FinalizationOutput:
        context.checkpoint()
        if not inputs.scenes:
            raise ValueError("Finalization requires verified scene artifacts")
        scene_ids = [scene.plan.scene_id for scene in inputs.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Finalization contains duplicate scene identities")
        videos = [
            validate_artifact(inputs.root, inputs.run_id, scene, inputs.render_profile)
            for scene in sorted(inputs.scenes, key=lambda item: item.plan.scene_id)
        ]
        requested = inputs.output_path.expanduser().absolute()
        if requested.is_symlink() or requested.resolve() in videos:
            raise ValueError("Final output must not alias an input artifact")
        with inputs.config.activate():
            # VideoMerger owns the output lock, ffprobe verification and atomic replace.
            merged = Path(
                self.merger.merge(
                    videos,
                    requested,
                    render_profile=inputs.render_profile,
                    merge_profile=inputs.merge_profile,
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

    def evaluate(
        self, inputs: FinalizationInputs, video: FinalizationOutput, context: ToolContext
    ) -> FinalizationOutput:
        context.checkpoint()
        settings = inputs.config.resolve()
        if settings.ENABLE_VISUAL_EVAL or settings.ENABLE_AUTO_EVAL:
            raise NotImplementedError(
                "视觉/visual evaluation is unsupported in the single-scene runtime"
            )
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
        return video
