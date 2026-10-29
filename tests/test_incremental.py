"""Incremental reuse binds exact code, render identity and self-contained video evidence."""

import shutil
import time
from dataclasses import replace

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_cross_module import scene_inputs

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.application import Application
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.incremental import (
    IncrementalInputs,
    IncrementalPipeline,
    validate_reused_artifact,
)
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import RunRepository, get_reusable_video_path, sha256_text
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification

NEW_RUN_ID = "20261029-120000-deadbeef"
GOAL = "Show a one-second pause"


def incremental_inputs(agent_boundary):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    assert app.generate(GOAL, approve_plan=True) is not None
    base_root = next((boundary.workspace / "runs").iterdir())
    base = RunRepository(boundary.workspace).load_for_resume(base_root.name)
    current_root = boundary.workspace / "runs" / NEW_RUN_ID
    current_root.mkdir(mode=0o700)
    scene = base.scenes[1].model_copy(deep=True)
    code = (base_root / scene.code_file).read_text()
    scene.code_file = ObjectStore(current_root).source(1, code)
    scene.slurm_job = scene.submission = scene.artifact = None
    scene.rendered = False
    scene.phase = "reviewed"
    scene.execution_verification = ExecutionVerification()
    scene.visual_verification = VisualVerification()
    current = replace(
        scene_inputs(base, base_root), run_id=NEW_RUN_ID, root=current_root, scene=scene
    )
    return IncrementalInputs(
        base=base,
        base_root=base_root,
        current=current,
        generation_identity=base.agent.generation_identity,
        current_scenes=(scene,),
        rag_profile=base.rag_profile,
        continuity_ready=True,
        backend=base.backend,
    )


def tool_context():
    return ToolContext(
        deadline=time.monotonic() + 30,
        cancelled=lambda: False,
        record_external_id=lambda external_id: pytest.fail("reuse dispatched a render"),
    )


def test_public_incremental_reuses_verified_matching_artifact(agent_boundary):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    assert app.generate(GOAL, approve_plan=True) is not None
    base_root = next((boundary.workspace / "runs").iterdir())
    before = (base_root / "manifest.json").read_bytes()
    published = []

    def require_exact_receipt(manifest):
        for scene in manifest.scenes.values():
            if scene.rendered:
                assert scene.artifact is not None
                assert scene.execution_verification.status == "passed"
                assert scene.execution_verification.artifact_sha256 == scene.artifact.video_sha256
                published.append(scene.artifact.origin)

    dependencies.after_commit = require_exact_receipt
    output = app.run_incremental(GOAL, base_root.name, approve_plan=True)
    assert output is not None
    root = output.parent
    saved = RunRepository(boundary.workspace).load_for_resume(root.name)
    scene = saved.scenes[1]
    assert scene.rendered and scene.slurm_job is None and scene.submission is None
    assert scene.artifact.origin == "reused" and scene.artifact.source_run_id == root.name
    assert "reused" in published and len(boundary.backend.sources) == 1
    assert (
        get_reusable_video_path(saved, 1, root).read_bytes()
        == (
            get_reusable_video_path(
                RunRepository(boundary.workspace).load(base_root.name), 1, base_root
            )
        ).read_bytes()
    )
    assert (base_root / "manifest.json").read_bytes() == before


def test_incremental_copy_is_self_contained_after_base_cleanup(agent_boundary):
    inputs = incremental_inputs(agent_boundary)
    before = inputs.base.model_dump_json()
    result = IncrementalPipeline().reuse(inputs, tool_context())
    assert result.eligible and result.scene.artifact.origin == "reused"
    assert inputs.base.model_dump_json() == before
    assert not inputs.current.scene.rendered and inputs.current.scene.artifact is None
    shutil.rmtree(inputs.base_root)
    copied = validate_reused_artifact(
        inputs.current.root,
        inputs.current.run_id,
        result.scene,
        inputs.current.render_profile,
        provenance=result.provenance,
    )
    assert sha256_file(copied) == result.scene.artifact.video_sha256
    assert copied.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", ["code", "render_profile"])
def test_incremental_rejects_changed_scene_inputs(agent_boundary, change):
    inputs = incremental_inputs(agent_boundary)
    current = inputs.current
    if change == "code":
        scene = current.scene.model_copy(deep=True)
        code = (current.root / scene.code_file).read_text().replace("self.wait(1)", "self.wait(2)")
        scene.code_file = ObjectStore(current.root).source(1, code)
        scene.code_sha256 = sha256_text(code)
        scene.static_verification = StaticVerification(
            status="passed",
            code_sha256=scene.code_sha256,
            technical_spec_sha256=scene.technical_spec_sha256,
        )
        current = replace(current, scene=scene)
        inputs = replace(inputs, current=current, current_scenes=(scene,))
    else:
        profile = current.render_profile.model_copy(
            update={"frame_rate": current.render_profile.frame_rate + 1}
        )
        current = replace(
            current,
            render_profile=profile,
            execution_spec=current.execution_spec.for_profile(profile),
        )
        inputs = replace(inputs, current=current)
    original = (inputs.base_root / inputs.base.scenes[1].artifact.video_path).read_bytes()
    result = IncrementalPipeline().reuse(inputs, tool_context())
    assert not result.eligible and result.scene is None and result.video is None
    assert ("code_changed" if change == "code" else "render_changed") in result.reasons
    assert not list((current.root / "videos").glob("**/*.mp4"))
    assert (inputs.base_root / inputs.base.scenes[1].artifact.video_path).read_bytes() == original


@pytest.mark.parametrize("change", ["missing", "tool_version", "execution_digest", "policy"])
def test_incomplete_or_different_environment_refuses_reuse_and_preserves_video(
    agent_boundary, change
):
    inputs = incremental_inputs(agent_boundary)
    video = inputs.base_root / inputs.base.scenes[1].artifact.video_path
    original = video.read_bytes()
    artifact = inputs.base.scenes[1].artifact
    if change == "missing":
        artifact.environment_fingerprint.clear()
    elif change == "tool_version":
        artifact.environment_fingerprint["manim"] = "different"
    elif change == "execution_digest":
        artifact.environment_fingerprint["execution_sha256"] = "b" * 64
    else:
        spec = inputs.current.execution_spec
        inputs = replace(
            inputs,
            current=replace(
                inputs.current,
                execution_spec=spec.model_copy(
                    update={"disable_network": not spec.disable_network}
                ),
            ),
        )
    with pytest.raises(ValueError):
        IncrementalPipeline().reuse(inputs, tool_context())
    assert video.read_bytes() == original
    assert not list((inputs.current.root / "videos").glob("**/*.mp4"))


def test_incremental_rejects_tampered_video_without_copying(agent_boundary):
    inputs = incremental_inputs(agent_boundary)
    video = inputs.base_root / inputs.base.scenes[1].artifact.video_path
    video.write_bytes(b"tampered base video")
    with pytest.raises(ValueError):
        IncrementalPipeline().reuse(inputs, tool_context())
    assert video.read_bytes() == b"tampered base video"
    assert not list((inputs.current.root / "videos").glob("**/*.mp4"))
