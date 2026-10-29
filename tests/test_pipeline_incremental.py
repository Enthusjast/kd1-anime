"""Incremental stages reuse exact private bytes, never jobs or mutable aliases."""

import shutil
import time
from dataclasses import replace

import pytest

from kd1_anime.agent_runtime.contracts import ObjectRef
from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.agent_runtime.tools import ToolContext
from kd1_anime.config import settings
from kd1_anime.domain.dependencies import SceneDependency
from kd1_anime.domain.planning import LessonSpec, ScenePlan, TeachingGraph
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig, VisualEvalProfile
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.pipeline.incremental import (
    IncrementalInputs,
    IncrementalOutput,
    IncrementalPipeline,
    ReuseProvenance,
    validate_reused_artifact,
)
from kd1_anime.prompt_identity import GenerationIdentity
from kd1_anime.rag.models import RagRuntimeProfile
from kd1_anime.rendering import RenderProfile, SceneArtifact, VideoMetadata, sha256_file
from kd1_anime.run_store import RunManifest, StoredSceneState, StoredSlurmJob, sha256_text
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification

BASE_ID = "20261027-120000-aabbccdd"
CURRENT_ID = "20261027-120001-11223344"
SOURCE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"


def plan(scene_id=1):
    return ScenePlan(
        scene_id=scene_id,
        title="Pause",
        duration_seconds=1,
        purpose="Pause",
        math_concept="Time",
        visual_design="Empty frame",
        camera_movement="Fixed",
        visual_flow=["Wait"],
        key_moments=["Pause"],
        computation="None",
        dependencies_complete=True,
    )


def context():
    return ToolContext(
        deadline=time.monotonic() + 60,
        cancelled=lambda: False,
        record_external_id=lambda _: pytest.fail("Reuse cannot publish a scheduler ID"),
    )


@pytest.fixture
def incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SLURM_REQUIRE_CONTAINER", False)
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", None)
    profile = RenderProfile(
        renderer="cairo",
        quality="l",
        pixel_width=854,
        pixel_height=480,
        frame_rate=15,
        manim_version="fixture manim",
        ffmpeg_version="fixture ffmpeg",
        xelatex_version="fixture xelatex",
    )
    spec = ExecutionSpec.capture(profile)
    config = RunConfig.capture()
    identity = GenerationIdentity(
        **{name: "a" * 64 for name in GenerationIdentity.model_fields if name != "format_version"}
    )
    base_root, root = tmp_path / BASE_ID, tmp_path / CURRENT_ID
    base_root.mkdir(mode=0o700)
    root.mkdir(mode=0o700)
    base_source = ObjectStore(base_root).source(1, SOURCE)
    current_source = ObjectStore(root).source(1, SOURCE)
    video = base_root / "videos/scene_1/attempt_original/Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"exact verified media bytes")
    final = base_root / "output.mp4"
    final.write_bytes(b"exact merged output")
    final_sha256 = sha256_file(final)
    private_final = base_root / "artifacts/videos" / f"{final_sha256}.mp4"
    private_final.parent.mkdir(parents=True)
    private_final.write_bytes(final.read_bytes())
    metadata = VideoMetadata(
        size_bytes=video.stat().st_size,
        duration_seconds=1,
        width=854,
        height=480,
        frame_rate=15,
    )
    environment = {
        "execution_sha256": spec.digest(),
        "python": spec.python_version,
        "manim": profile.manim_version,
        "ffmpeg": profile.ffmpeg_version,
        "xelatex": profile.xelatex_version,
        "renderer": profile.renderer,
    }
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=BASE_ID,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(SOURCE),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(base_root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=metadata,
        execution_sha256=spec.digest(),
        environment_fingerprint=environment,
    )
    job = StoredSlurmJob(
        job_id="123",
        scene_id=1,
        script_path="scenes/Demo.sbatch",
        log_out="logs/123.out",
        log_err="logs/123.err",
        media_dir="videos/scene_1/attempt_original",
        scene_class_name="Demo",
        submitted_at=1,
        status="COMPLETED",
        code_sha256=artifact.code_sha256,
        render_profile=profile,
        execution_sha256=spec.digest(),
        output_path=artifact.video_path,
        output_sha256=artifact.video_sha256,
        output_metadata=metadata,
        environment_fingerprint=environment,
    )
    static = StaticVerification(
        status="passed", code_sha256=artifact.code_sha256, checked_at="current review"
    )
    old_scene = StoredSceneState(
        plan=plan(),
        revision=4,
        input_sha256="b" * 64,
        code_file=base_source,
        code_sha256=artifact.code_sha256,
        class_name="Demo",
        reviewed=True,
        rendered=True,
        artifact=artifact,
        slurm_job=job,
        static_verification=static,
        execution_verification=ExecutionVerification(
            status="passed",
            scope="formal_video",
            code_sha256=artifact.code_sha256,
            artifact_sha256=artifact.video_sha256,
            duration_seconds=1,
            checked_at="original execution",
        ),
    )
    scene = old_scene.model_copy(
        deep=True,
        update={
            "revision": 0,
            "input_sha256": "c" * 64,
            "code_file": current_source,
            "rendered": False,
            "artifact": None,
            "slurm_job": None,
            "execution_verification": ExecutionVerification(),
        },
    )
    base = RunManifest(
        schema_version=11,
        run_id=BASE_ID,
        user_prompt="base",
        state="DONE",
        status="completed",
        output_path=str(final),
        final_video="output.mp4",
        final_video_sha256=sha256_file(final),
        run_config=config,
        execution_spec=spec,
        render_profile=profile,
        scenes={1: old_scene},
        artifact_refs={"final_video": private_final.relative_to(base_root).as_posix()},
        agent=AgentRunState(
            session_id="base-session", generation_identity=identity, run_input_sha256="b" * 64
        ),
    )
    (base_root / "manifest.json").write_text(base.model_dump_json(), encoding="utf-8")
    current = SceneInputs(
        CURRENT_ID,
        root,
        config,
        profile,
        spec,
        scene,
        (scene.plan,),
        LessonSpec(),
        TeachingGraph(),
        None,
    )
    return IncrementalInputs(
        base, base_root, current, identity, (scene,), RagRuntimeProfile(), continuity_ready=True
    )


def replace_scene(inputs, **updates):
    scene = inputs.current.scene.model_copy(deep=True, update=updates)
    current = replace(
        inputs.current,
        scene=scene,
        plans=tuple(
            scene.plan if item.scene_id == scene.plan.scene_id else item
            for item in inputs.current.plans
        ),
    )
    scenes = tuple(
        scene if item.plan.scene_id == scene.plan.scene_id else item
        for item in inputs.current_scenes
    )
    return replace(inputs, current=current, current_scenes=scenes)


def test_reuse_copies_exact_content_and_preserves_base(incremental):
    before = incremental.base.model_dump_json()
    disk_before = (incremental.base_root / "manifest.json").read_bytes()
    result = IncrementalPipeline().reuse(incremental, context())
    assert result.eligible and not result.reasons
    assert (
        result.scene.rendered and result.scene.slurm_job is None and result.scene.submission is None
    )
    assert result.scene.artifact.origin == "reused" and result.scene.artifact.job_id is None
    assert result.scene.artifact.source_run_id == CURRENT_ID
    copied = incremental.current.root / result.video.path
    original = incremental.base_root / incremental.base.scenes[1].artifact.video_path
    assert copied.read_bytes() == original.read_bytes()
    assert copied.stat().st_ino != original.stat().st_ino
    assert copied.stat().st_mode & 0o777 == 0o600
    assert copied.parent.stat().st_mode & 0o777 == 0o700
    assert result.scene.static_verification == incremental.current.scene.static_verification
    assert result.scene.execution_verification.checked_at == "original execution"
    assert result.scene.execution_verification.artifact_sha256 == result.video.sha256
    assert incremental.base.model_dump_json() == before
    assert (incremental.base_root / "manifest.json").read_bytes() == disk_before
    assert not incremental.current.scene.rendered
    snapshot = incremental.current.root / result.snapshot.path
    assert sha256_file(snapshot) == result.snapshot.sha256
    assert StoredSceneState.model_validate_json(snapshot.read_text()) == result.scene
    assert IncrementalOutput.model_validate_json(result.model_dump_json()) == result


def test_current_run_validates_without_base_after_cleanup(incremental):
    result = IncrementalPipeline().reuse(incremental, context())
    shutil.rmtree(incremental.base_root)
    path = validate_reused_artifact(
        incremental.current.root,
        CURRENT_ID,
        result.scene,
        incremental.current.render_profile,
        provenance=result.provenance,
    )
    assert sha256_file(path) == result.video.sha256
    receipt = ReuseProvenance.model_validate_json(
        (incremental.current.root / result.provenance.path).read_text()
    )
    assert receipt.base_run_id == BASE_ID and receipt.base_artifact.job_id == "123"


@pytest.mark.parametrize("change", ["code", "plan", "profile", "inherited", "technical", "global"])
def test_legitimate_changed_inputs_return_no_artifact(incremental, change):
    inputs = incremental
    if change == "code":
        source = SOURCE.replace("self.wait()", "self.wait(2)")
        path = ObjectStore(inputs.current.root).source(1, source)
        static = StaticVerification(status="passed", code_sha256=sha256_text(source))
        inputs = replace_scene(
            inputs, code_file=path, code_sha256=sha256_text(source), static_verification=static
        )
    elif change == "plan":
        inputs = replace_scene(
            inputs, plan=inputs.current.scene.plan.model_copy(update={"title": "Changed"})
        )
    elif change == "profile":
        profile = inputs.current.render_profile.model_copy(update={"frame_rate": 30})
        inputs = replace(
            inputs,
            current=replace(
                inputs.current,
                render_profile=profile,
                execution_spec=inputs.current.execution_spec.for_profile(profile),
            ),
        )
    elif change == "inherited":
        inputs = replace_scene(inputs, inherited_elements_code="# different inherited context")
    elif change == "technical":
        technical = TechnicalSpec(scene_id=1)
        technical_hash = sha256_text(technical.model_dump_json())
        static = inputs.current.scene.static_verification.model_copy(
            update={"technical_spec_sha256": technical_hash}
        )
        inputs = replace_scene(
            inputs,
            technical_spec=technical,
            technical_spec_sha256=technical_hash,
            static_verification=static,
        )
    else:
        inputs = replace(
            inputs, current=replace(inputs.current, lesson_spec=LessonSpec(topic="Changed lesson"))
        )
    result = IncrementalPipeline().reuse(inputs, context())
    assert not result.eligible and result.reasons
    assert result.scene is result.snapshot is result.video is result.provenance is None
    assert not (inputs.current.root / "videos").exists()


@pytest.mark.parametrize("identity", ["generation", "execution", "backend"])
def test_incompatible_identity_fails_before_new_run_write(incremental, identity):
    inputs = incremental
    if identity == "generation":
        inputs = replace(
            inputs,
            generation_identity=inputs.generation_identity.model_copy(
                update={"policy_sha256": "f" * 64}
            ),
        )
    elif identity == "execution":
        inputs = replace(
            inputs,
            current=replace(
                inputs.current,
                execution_spec=inputs.current.execution_spec.model_copy(
                    update={"disable_network": not inputs.current.execution_spec.disable_network}
                ),
            ),
        )
    else:
        inputs = replace(inputs, backend="local")
    with pytest.raises(ValueError):
        IncrementalPipeline().reuse(inputs, context())
    assert not (inputs.current.root / "artifacts").exists()


@pytest.mark.parametrize("tamper", ["video", "source", "final", "environment", "alias"])
def test_base_corruption_fails_closed(incremental, tamper):
    base_scene = incremental.base.scenes[1]
    if tamper == "video":
        (incremental.base_root / base_scene.artifact.video_path).write_bytes(b"tampered")
    elif tamper == "source":
        (incremental.base_root / base_scene.code_file).write_text("tampered")
    elif tamper == "final":
        (incremental.base_root / "output.mp4").write_bytes(b"tampered")
    elif tamper == "environment":
        base_scene.artifact.environment_fingerprint["python"] = "different Python"
    else:
        alias = incremental.base_root / "scenes/scene_1.py"
        alias.write_text(SOURCE)
        base_scene.code_file = "scenes/scene_1.py"
    with pytest.raises((ValueError, RuntimeError)):
        IncrementalPipeline().reuse(incremental, context())
    assert not (incremental.current.root / "artifacts").exists()


def test_reuse_invalidates_visual_evidence_for_current_context(incremental):
    old = incremental.current.scene
    inputs = replace_scene(
        incremental,
        visual_verification=VisualVerification(status="passed", score=5),
        visual_report_file="old-report.json",
        visual_report_sha256="f" * 64,
        visual_artifact_sha256="f" * 64,
        visual_feedback="old feedback",
        visual_score=5,
    )
    inputs = replace(
        inputs, current=replace(inputs.current, visual_eval_profile=VisualEvalProfile(enabled=True))
    )
    result = IncrementalPipeline().reuse(inputs, context())
    assert result.scene.visual_status == "pending"
    assert result.scene.visual_verification.status == "not_run"
    assert result.scene.visual_score is None
    assert not result.scene.visual_report_file and not result.scene.visual_artifact_sha256
    assert old.visual_status == "skipped"


@pytest.mark.parametrize(
    "tamper", ["report", "reference", "video", "scene", "job", "input_digest", "source", "context"]
)
def test_reused_validator_requires_exact_typed_provenance(incremental, tamper):
    result = IncrementalPipeline().reuse(incremental, context())
    scene = result.scene.model_copy(deep=True)
    reference = result.provenance
    if tamper == "report":
        (incremental.current.root / reference.path).write_text("{}")
    elif tamper == "reference":
        reference = ObjectRef(
            path="artifacts/objects/" + "0" * 64 + ".json", sha256="0" * 64, kind="report"
        )
    elif tamper == "video":
        (incremental.current.root / result.video.path).write_bytes(b"tampered")
    elif tamper == "scene":
        scene.plan.title = "Changed"
    elif tamper == "input_digest":
        scene.input_sha256 = "d" * 64
    elif tamper == "source":
        scene.code_sha256 = "d" * 64
    elif tamper == "context":
        scene.inherited_elements_code = "# Changed inherited context"
    else:
        scene.artifact.job_id = "123"
    with pytest.raises(ValueError):
        validate_reused_artifact(
            incremental.current.root,
            CURRENT_ID,
            scene,
            incremental.current.render_profile,
            provenance=reference,
        )


def test_reuse_requires_review_and_ordered_barrier(incremental):
    with pytest.raises(ValueError, match="continuity barrier"):
        IncrementalPipeline().reuse(replace(incremental, continuity_ready=False), context())
    with pytest.raises(ValueError, match="reviewed static"):
        IncrementalPipeline().reuse(replace_scene(incremental, reviewed=False), context())


@pytest.mark.parametrize("kind", ["teaching", "handoff", "code_context"])
def test_dependency_specific_changes_invalidate_consumer(incremental, kind):
    producer = incremental.base.scenes[1].model_copy(deep=True)
    consumer = producer.model_copy(deep=True)
    consumer.plan = plan(2).model_copy(
        update={"dependencies": [SceneDependency(source_scene_id=1, kind=kind)]}
    )
    consumer.code_file = ObjectStore(incremental.base_root).source(2, SOURCE)
    consumer.artifact.scene_id = consumer.slurm_job.scene_id = 2
    # Both verified scene records may reference the same exact completed bytes.
    incremental.base.scenes[2] = consumer
    current_consumer = consumer.model_copy(
        deep=True,
        update={
            "code_file": ObjectStore(incremental.current.root).source(2, SOURCE),
            "rendered": False,
            "artifact": None,
            "slurm_job": None,
            "execution_verification": ExecutionVerification(),
        },
    )
    if kind == "teaching":
        producer.plan.purpose = "Different teaching"
    elif kind == "handoff":
        producer.technical_spec = TechnicalSpec(scene_id=1, export_element_ids=["different"])
        # Contract digest only binds exported object values, so add explicit handoff evidence.
        from kd1_anime.domain.technical import TechnicalHandoff

        producer.technical_spec.handoff_out = TechnicalHandoff(source_scene_id=1)
    else:
        producer.code_sha256 = "f" * 64
    current = replace(
        incremental.current,
        scene=current_consumer,
        plans=(producer.plan, current_consumer.plan),
    )
    inputs = replace(incremental, current=current, current_scenes=(producer, current_consumer))
    result = IncrementalPipeline().reuse(inputs, context())
    assert not result.eligible and "dependencies_changed" in result.reasons


def test_teaching_dependency_does_not_invalidate_on_unrelated_source_change(incremental):
    producer = incremental.base.scenes[1].model_copy(deep=True)
    consumer = producer.model_copy(deep=True)
    consumer.plan = plan(2).model_copy(
        update={"dependencies": [SceneDependency(source_scene_id=1, kind="teaching")]}
    )
    consumer.code_file = ObjectStore(incremental.base_root).source(2, SOURCE)
    consumer.artifact.scene_id = consumer.slurm_job.scene_id = 2
    incremental.base.scenes[2] = consumer
    scene = consumer.model_copy(
        deep=True,
        update={
            "code_file": ObjectStore(incremental.current.root).source(2, SOURCE),
            "rendered": False,
            "artifact": None,
            "slurm_job": None,
            "execution_verification": ExecutionVerification(),
        },
    )
    producer.code_sha256 = "f" * 64
    inputs = replace(
        incremental,
        current=replace(incremental.current, scene=scene, plans=(producer.plan, scene.plan)),
        current_scenes=(producer, scene),
    )
    assert IncrementalPipeline().reuse(inputs, context()).eligible


def test_identical_reuse_reopens_only_exact_immutable_objects(incremental):
    first = IncrementalPipeline().reuse(incremental, context())
    second = IncrementalPipeline().reuse(incremental, context())
    assert first == second
    assert not incremental.current.scene.rendered


@pytest.mark.parametrize("schema_version", range(4, 11))
def test_readonly_base_version_rejected_before_copy(incremental, schema_version):
    base = incremental.base.model_copy(update={"schema_version": schema_version, "agent": None})
    with pytest.raises(ValueError, match="schema_version"):
        IncrementalPipeline().reuse(replace(incremental, base=base), context())
    assert not (incremental.current.root / "artifacts").exists()


def test_reused_validator_never_authorizes_from_path_alone(incremental):
    result = IncrementalPipeline().reuse(incremental, context())
    with pytest.raises(ValueError, match="typed provenance"):
        validate_reused_artifact(
            incremental.current.root,
            CURRENT_ID,
            result.scene,
            incremental.current.render_profile,
            provenance=None,
        )


def test_new_scene_has_no_fake_artifact(incremental):
    source = ObjectStore(incremental.current.root).source(2, SOURCE)
    scene = incremental.current.scene.model_copy(
        deep=True,
        update={
            "plan": plan(2),
            "code_file": source,
        },
    )
    current = replace(incremental.current, scene=scene, plans=(scene.plan,))
    result = IncrementalPipeline().reuse(
        replace(incremental, current=current, current_scenes=(scene,)), context()
    )
    assert not result.eligible and result.reasons == ("scene_missing",)
    assert result.scene is None and result.video is None


@pytest.mark.parametrize("revision_advance", [1, 4])
def test_reuse_identity_survives_nonsemantic_commit_revisions(incremental, revision_advance):
    result = IncrementalPipeline().reuse(incremental, context())
    scene = result.scene.model_copy(
        deep=True,
        update={
            "revision": result.scene.revision + revision_advance,
        },
    )
    receipt = ReuseProvenance.model_validate_json(
        (incremental.current.root / result.provenance.path).read_text()
    )
    assert receipt.work_identity.revision == result.scene.revision
    assert receipt.work_identity.input_sha256 == scene.input_sha256
    assert receipt.input_sha256 == result.input_sha256
    assert receipt.work_identity.input_sha256 != receipt.input_sha256
    path = validate_reused_artifact(
        incremental.current.root,
        CURRENT_ID,
        scene,
        incremental.current.render_profile,
        provenance=result.provenance,
    )
    assert sha256_file(path) == result.video.sha256


def test_reuse_requires_full_semantic_input_digest(incremental):
    inputs = replace_scene(incremental, input_sha256="")
    with pytest.raises(ValueError, match="full input digest"):
        IncrementalPipeline().reuse(inputs, context())
    assert not (inputs.current.root / "artifacts").exists()


def test_reuse_identity_rejects_a_scene_older_than_dispatch(incremental):
    inputs = replace_scene(incremental, revision=3)
    result = IncrementalPipeline().reuse(inputs, context())
    scene = result.scene.model_copy(deep=True, update={"revision": 2})
    with pytest.raises(ValueError, match="provenance differs"):
        validate_reused_artifact(
            inputs.current.root,
            CURRENT_ID,
            scene,
            inputs.current.render_profile,
            provenance=result.provenance,
        )


def external_completed_output(inputs):
    external = inputs.base_root.parent / "exported.mp4"
    external.write_bytes((inputs.base_root / "output.mp4").read_bytes())
    inputs.base.output_path = inputs.base.final_video = str(external)
    return external


def test_completed_external_output_and_private_snapshot_are_reusable(incremental):
    external = external_completed_output(incremental)
    before = incremental.base.model_dump_json()
    result = IncrementalPipeline().reuse(incremental, context())
    assert result.eligible and external.is_file()
    assert incremental.base.model_dump_json() == before


@pytest.mark.parametrize("tamper", ["public", "private", "symlink", "wrong_output"])
def test_completed_external_output_corruption_fails_before_copy(incremental, tamper):
    external = external_completed_output(incremental)
    if tamper == "public":
        external.write_bytes(b"tampered public output")
    elif tamper == "private":
        private = incremental.base_root / incremental.base.artifact_refs["final_video"]
        private.write_bytes(b"tampered immutable output")
    elif tamper == "symlink":
        external.unlink()
        external.symlink_to(incremental.base_root / "output.mp4")
    else:
        wrong = external.with_name("different.mp4")
        wrong.write_bytes(external.read_bytes())
        incremental.base.final_video = str(wrong)
    with pytest.raises(RuntimeError):
        IncrementalPipeline().reuse(incremental, context())
    assert not (incremental.current.root / "artifacts").exists()
