"""Consumer contracts against the real workflow and deterministic boundary collaborators.

No generated Python, model endpoint, scheduler command, or media executable runs here.
The existing autouse endpoint-probe isolation also applies to these tests.
"""

from __future__ import annotations

import json
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import kd1_anime.orchestrator as workflow
from kd1_anime.application import Application
from kd1_anime.cluster.jobs import PollObservation, RenderJob
from kd1_anime.config import settings
from kd1_anime.domain.plan_review import ContinuityReviewResult, PlanReviewResult
from kd1_anime.domain.planning import ContinuityBible, SceneOutline, ScenePlan
from kd1_anime.domain.review import ReviewResult
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.rendering import RenderProfile, VideoMetadata, sha256_file
from kd1_anime.request_budget import RequestBudgetExceeded
from kd1_anime.run_store import RunRepository, sha256_text
from kd1_anime.services.rendering import RenderingService

SOURCE = (
    "from manim import *\nclass Pause(Scene):\n    def construct(self):\n        self.wait(1)\n"
)
VIDEO = b"deterministic-backend-video"


class ArtifactBackend:
    """Own the render boundary without importing or executing submitted source."""

    name = "slurm"

    def __init__(self):
        self.sources: list[Path] = []
        self.jobs: dict[str, RenderJob] = {}

    def submit_scene(
        self,
        scene_id,
        python_file,
        scene_class_name="Scene",
        *,
        scenes_dir,
        logs_dir,
        videos_dir,
        code_sha256,
        render_profile,
        resource_profile=None,
        attempt_token=None,
    ):
        source = Path(python_file)
        assert source.read_text(encoding="utf-8") == SOURCE
        assert sha256_file(source) == code_sha256
        self.sources.append(source)
        job_id = str(len(self.jobs) + 1)
        job = RenderJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=scenes_dir / f"render_{scene_id}.sh",
            log_out=logs_dir / f"scene_{scene_id}.out",
            log_err=logs_dir / f"scene_{scene_id}.err",
            media_dir=videos_dir / f"scene_{scene_id}" / f"attempt_{attempt_token}",
            scene_class_name=scene_class_name,
            submitted_at=time.time(),
            execution_sha256=self.execution_spec.digest(),
            code_sha256=code_sha256,
            render_profile=render_profile,
            resource_profile=resource_profile,
        )
        self.jobs[job_id] = job
        return job

    def poll(self, job_ids):
        return PollObservation({job_id: "COMPLETED" for job_id in job_ids}, {})

    def forward_log(self, job, positions):
        return None

    def classify_gone(self, job):
        return None

    def validate_completed_job(self, job):
        job.media_dir.mkdir(parents=True, exist_ok=True)
        job.output_path = job.media_dir / f"{job.scene_class_name}.mp4"
        job.output_path.write_bytes(VIDEO)
        job.output_sha256 = sha256_file(job.output_path)
        job.output_metadata = VideoMetadata(
            size_bytes=len(VIDEO),
            duration_seconds=1,
            width=job.render_profile.pixel_width,
            height=job.render_profile.pixel_height,
            frame_rate=job.render_profile.frame_rate,
        )
        job.environment_fingerprint = {
            "manim": job.render_profile.manim_version,
            "ffmpeg": job.render_profile.ffmpeg_version,
            "xelatex": job.render_profile.xelatex_version,
            "renderer": job.render_profile.renderer,
            "pyopengl_platform": job.render_profile.opengl_platform,
            "python": self.execution_spec.python_version,
            "execution_sha256": self.execution_spec.digest(),
        }
        return True

    def cancel_job(self, job_id):
        return True

    def get_error_log(self, *args, **kwargs):
        return ""


class ArtifactMerger:
    def merge(self, videos, output, **kwargs):
        assert len(videos) == 1
        assert Path(videos[0]).read_bytes() == VIDEO
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"merged:" + VIDEO)
        return output


class IsolatedRenderingService(RenderingService):
    """The fake backend has no host Slurm/Manim installation to preflight."""

    def preflight(self, profile):
        return None


@pytest.fixture
def application_boundary(monkeypatch, tmp_path):
    for name, value in {
        "WORKSPACE_DIR": tmp_path,
        "LLM_API_KEY": "application-test-key",
        "LLM_MODEL": "deterministic-model",
        "RAG_ENABLED": False,
        "RAG_RECIPES_DIR": tmp_path / "recipes",
        "FAILURE_CASES_PATH": tmp_path / "failure-cases.sqlite3",
        "ENABLE_VISUAL_EVAL": False,
        "ENABLE_AUTO_EVAL": False,
        "LOCAL_SMOKE_RENDER_ENABLED": False,
        "SLURM_REQUIRE_CONTAINER": False,
        "SLURM_CONTAINER_IMAGE": None,
        "RENDER_BACKEND": "slurm",
        "CODEGEN_MODE": "python",
        "SCENE_CODEGEN_MODE": "llm",
        "GENERATION_MODE": "strict",
        "MAX_PLAN_REVIEW_ROUNDS": 1,
        "MAX_PLAN_REPLAN_ATTEMPTS": 1,
        "MAX_CONTINUITY_FIX_ROUNDS": 0,
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(
        RenderProfile,
        "_environment_versions",
        staticmethod(lambda: ("0.20.1", "fixture ffmpeg", "fixture xelatex")),
    )
    backend = ArtifactBackend()
    calls = []
    rejecting = SimpleNamespace(plan=False)

    def reserve(agent, operation):
        """Act like an endpoint, requiring its reservation to exist before dispatch."""
        receipt = agent.request_context.reserve("main", "deterministic-model")
        manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (tmp_path / "runs").glob("*/manifest.json")
        ]
        assert any(
            item["request_id"] == receipt.request_id
            for manifest in manifests
            for item in manifest["request_budget"]["reservations"]
        )
        calls.append(operation)
        agent.request_context.complete(receipt, time.monotonic())

    class Planner:
        def plan_outline(self, prompt, **kwargs):
            reserve(self, "outline")
            return [
                SceneOutline(
                    scene_id=1,
                    title="Pause",
                    duration_seconds=1,
                    purpose="Show a pause",
                    math_concept="Time",
                )
            ]

        def plan_continuity_bible(self, prompt, outlines, **kwargs):
            reserve(self, "bible")
            return ContinuityBible()

        def plan_detail(self, outline, all_outlines, prompt, **kwargs):
            reserve(self, "detail")
            return ScenePlan(
                scene_id=outline.scene_id,
                title=outline.title,
                duration_seconds=outline.duration_seconds,
                purpose=outline.purpose,
                math_concept=outline.math_concept,
                visual_design="An empty frame remains on screen for one second",
                camera_movement="Fixed camera",
                visual_flow=["Wait one second"],
                key_moments=["One second elapses"],
                computation="No computation",
                opening_state=["Empty frame"],
                closing_state=["Empty frame"],
                transition_in="Start with empty frame",
                transition_out="Keep empty frame",
            )

    class PlanReviewer:
        def review(self, *args, **kwargs):
            reserve(self, "plan_review")
            if rejecting.plan:
                return PlanReviewResult(
                    is_valid=False,
                    severity="major",
                    summary="The requested pause is missing",
                    issues=[
                        {
                            "category": "feasibility",
                            "severity": "major",
                            "field": "visual_flow",
                            "confidence": "high",
                            "evidence_type": "contract",
                            "evidence": "The consumer explicitly requires a one-second pause",
                            "message": "The requested pause is missing",
                            "fix_instruction": "Include an explicit one-second pause",
                        }
                    ],
                )
            return PlanReviewResult(is_valid=True, summary="The pause is implementable")

    class ContinuityReviewer:
        def review(self, *args, **kwargs):
            reserve(self, "continuity_review")
            return ContinuityReviewResult(is_valid=True, summary="Consistent frame")

    class TechnicalPlanner:
        def plan(self, scene_plan, *, renderer=None, **kwargs):
            reserve(self, "technical")
            return TechnicalSpec(scene_id=scene_plan.scene_id, renderer=renderer or "cairo")

    class Coder:
        def generate_code(self, scene_plan, **kwargs):
            reserve(self, "code")
            return SOURCE

    class Reviewer:
        def review(self, code, scene_plan, **kwargs):
            reserve(self, "code_review")
            return ReviewResult(is_valid=True)

    for name, collaborator in {
        "PlannerAgent": Planner,
        "PlanReviewerAgent": PlanReviewer,
        "ContinuityReviewerAgent": ContinuityReviewer,
        "TechnicalPlannerAgent": TechnicalPlanner,
        "CoderAgent": Coder,
        "ReviewerAgent": Reviewer,
        "VideoMerger": ArtifactMerger,
        "RenderingService": IsolatedRenderingService,
    }.items():
        monkeypatch.setattr(workflow, name, collaborator)
    monkeypatch.setattr(workflow, "create_render_backend", lambda *args, **kwargs: backend)
    return SimpleNamespace(
        app=Application(runtime_factory=workflow.Orchestrator),
        backend=backend,
        calls=calls,
        rejecting=rejecting,
        workspace=tmp_path,
    )


def saved_run(boundary):
    manifests = list((boundary.workspace / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    root = manifests[0].parent
    return root, RunRepository(boundary.workspace).load_for_resume(root.name)


def assert_source_receipt(root, scene):
    source = root / scene.code_file
    assert source.is_relative_to(root)
    assert source.read_text(encoding="utf-8") == SOURCE
    assert sha256_file(source) == scene.code_sha256 == sha256_text(SOURCE)
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    return source


def test_generate_publishes_video_manifest_events_and_request_receipts(application_boundary):
    boundary = application_boundary
    events = []
    output = boundary.workspace / "consumer.mp4"
    result = boundary.app.generate(
        "Show a one-second pause",
        output_path=output,
        callback=lambda name, data: events.append((name, data)),
    )
    assert result == output
    assert result.read_bytes() == b"merged:" + VIDEO
    root, manifest = saved_run(boundary)
    assert manifest.status == "completed"
    assert manifest.final_video_sha256 == sha256_file(result)
    assert manifest.user_prompt == "Show a one-second pause"
    scene = manifest.scenes[1]
    source = assert_source_receipt(root, scene)
    assert boundary.backend.sources == [source]
    assert scene.reviewed and scene.rendered
    assert scene.static_verification.status == "passed"
    assert scene.execution_verification.status == "passed"
    assert scene.artifact is not None
    artifact_path = root / scene.artifact.video_path
    assert artifact_path.read_bytes() == VIDEO
    assert scene.artifact.code_sha256 == scene.code_sha256
    assert scene.artifact.video_sha256 == sha256_file(artifact_path)
    assert manifest.artifact_refs
    assert all((root / relative).is_file() for relative in manifest.artifact_refs.values())
    requests = manifest.request_budget.reservations
    assert len(requests) == len(boundary.calls) > 0
    assert len({request.request_id for request in requests}) == len(requests)
    assert any(request.scene_id == 1 for request in requests)
    assert len(requests) <= manifest.request_budget.run_limit
    durable = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert all(event["run_id"] == manifest.run_id for event in durable)
    names = {event["event"] for event in durable}
    assert {"run_started", "scene_rendered", "merge_complete", "model_request_finished"} <= names
    assert {"run_started", "scene_rendered", "merge_complete"} <= {name for name, _ in events}
    assert all(event["manifest_revision"] <= manifest.revision for event in durable)


def test_dry_run_is_durably_complete_without_rendering(application_boundary):
    boundary = application_boundary
    assert boundary.app.generate("Show a one-second pause", dry_run=True) is None
    root, manifest = saved_run(boundary)
    assert manifest.status == "dry_run_complete"
    assert manifest.dry_run
    assert manifest.scenes[1].reviewed
    assert not manifest.scenes[1].rendered
    assert manifest.scenes[1].artifact is None
    assert manifest.final_video is None
    assert_source_receipt(root, manifest.scenes[1])
    assert not boundary.backend.sources
    previous_calls = list(boundary.calls)
    previous_manifest = (root / "manifest.json").read_bytes()
    assert Application(runtime_factory=workflow.Orchestrator).resume(manifest.run_id) is None
    assert boundary.calls == previous_calls
    assert (root / "manifest.json").read_bytes() == previous_manifest


@pytest.mark.parametrize("review", [True, False])
def test_plan_only_exports_committed_plan_and_never_generates_code(application_boundary, review):
    boundary = application_boundary
    export = boundary.workspace / "plan.json"
    plans = boundary.app.plan_only("Show a one-second pause", review=review, export_path=export)
    assert len(plans) == 1
    assert plans[0].title == "Pause"
    assert plans[0].visual_flow == ["Wait one second"]
    root, manifest = saved_run(boundary)
    payload = json.loads(export.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["user_prompt"] == manifest.user_prompt
    assert payload["items"] == [plan.model_dump(mode="json") for plan in plans]
    assert payload["lesson_spec"] == manifest.lesson_spec.model_dump(mode="json")
    assert payload["teaching_graph"] == manifest.teaching_graph.model_dump(mode="json")
    assert payload["continuity_bible"] == manifest.continuity_bible.model_dump(mode="json")
    assert manifest.status == "dry_run_complete"
    assert manifest.plan_review_status == ("passed" if review else "skipped")
    assert ("plan_review" in boundary.calls) is review
    assert not {"code", "technical", "code_review"}.intersection(boundary.calls)
    assert not manifest.scenes[1].code_file
    assert not boundary.backend.sources
    assert not list(root.rglob("*.py"))


def test_rejected_plan_does_not_publish_export_or_source(application_boundary):
    boundary = application_boundary
    boundary.rejecting.plan = True
    export = boundary.workspace / "rejected.json"
    with pytest.raises(RuntimeError, match="计划"):
        boundary.app.plan_only("Show a one-second pause", export_path=export)
    root, manifest = saved_run(boundary)
    assert manifest.status == "failed"
    assert not export.exists()
    assert not list(root.rglob("*.py"))
    assert not {"technical", "code", "code_review"}.intersection(boundary.calls)
    assert not boundary.backend.sources


def test_run_from_exported_plan_finishes_without_replanning_outline(application_boundary):
    boundary = application_boundary
    export = boundary.workspace / "plan.json"
    boundary.app.plan_only("Show a one-second pause", export_path=export)
    boundary.calls.clear()
    result = Application(runtime_factory=workflow.Orchestrator).run_from_plan(export)
    assert result is not None
    assert result.read_bytes() == b"merged:" + VIDEO
    assert "outline" not in boundary.calls
    assert "detail" not in boundary.calls
    assert "code" in boundary.calls
    manifests = [
        RunRepository(boundary.workspace).load_for_resume(path.parent.name)
        for path in (boundary.workspace / "runs").glob("*/manifest.json")
    ]
    assert sorted(manifest.status for manifest in manifests) == ["completed", "dry_run_complete"]


@pytest.mark.parametrize("damage", [None, "missing", "hash"])
def test_completed_resume_requires_original_final_video(application_boundary, damage):
    boundary = application_boundary
    result = boundary.app.generate("Show a one-second pause")
    root, manifest = saved_run(boundary)
    original_manifest = (root / "manifest.json").read_bytes()
    calls = list(boundary.calls)
    submissions = list(boundary.backend.sources)
    if damage == "missing":
        result.unlink()
    elif damage == "hash":
        result.write_bytes(b"tampered final video")
    resumed = Application(runtime_factory=workflow.Orchestrator)
    if damage is None:
        assert resumed.resume(manifest.run_id) == result
    else:
        with pytest.raises(RuntimeError, match="最终视频"):
            resumed.resume(manifest.run_id)
    assert boundary.calls == calls
    assert boundary.backend.sources == submissions
    assert (root / "manifest.json").read_bytes() == original_manifest


def test_direct_render_needs_no_llm_and_owns_private_source(application_boundary, monkeypatch):
    boundary = application_boundary
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    supplied = boundary.workspace / "supplied.py"
    supplied.write_text(SOURCE, encoding="utf-8")
    supplied.chmod(0o644)
    job, final_video, run_id = boundary.app.submit_existing_scene(
        supplied.read_text(encoding="utf-8"),
        "Pause",
        wait=True,
    )
    assert job.job_id == "1"
    assert final_video is not None
    assert final_video.read_bytes() == b"merged:" + VIDEO
    root, manifest = saved_run(boundary)
    assert manifest.run_id == run_id
    assert manifest.direct_render
    assert manifest.status == "completed"
    owned_source = assert_source_receipt(root, manifest.scenes[1])
    assert owned_source != supplied
    assert not owned_source.samefile(supplied)
    assert boundary.backend.sources == [owned_source]
    assert not boundary.calls
    assert not manifest.request_budget.reservations
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(supplied.stat().st_mode) == 0o644
    supplied.write_text("caller changed its source", encoding="utf-8")
    assert owned_source.read_text(encoding="utf-8") == SOURCE
    assert Application(runtime_factory=workflow.Orchestrator).resume(run_id) == final_video
    assert not boundary.calls


@pytest.mark.parametrize("entry", ["retry", "incremental"])
def test_unsafe_run_identifier_is_rejected_before_external_work(application_boundary, entry):
    boundary = application_boundary
    with pytest.raises((ValueError, FileNotFoundError)):
        if entry == "retry":
            boundary.app.retry("../outside", 1)
        else:
            boundary.app.run_incremental("Show a different pause", "../outside")
    assert not boundary.calls
    assert not boundary.backend.sources
    assert not list((boundary.workspace / "runs").glob("*/manifest.json"))


def test_completed_run_cannot_be_retried_in_place(application_boundary):
    boundary = application_boundary
    video = boundary.app.generate("Show a one-second pause")
    root, manifest = saved_run(boundary)
    before = (root / "manifest.json").read_bytes()
    calls = list(boundary.calls)
    with pytest.raises(RuntimeError):
        boundary.app.retry(manifest.run_id, 1)
    assert video.read_bytes() == b"merged:" + VIDEO
    assert (root / "manifest.json").read_bytes() == before
    assert boundary.calls == calls
    assert len(boundary.backend.sources) == 1


def test_request_budget_exhaustion_is_durable_and_blocks_generation(
    application_boundary, monkeypatch
):
    boundary = application_boundary
    monkeypatch.setattr(settings, "MAX_RUN_LLM_REQUESTS", 1)
    with pytest.raises((RequestBudgetExceeded, RuntimeError), match=r"预算|未能完成|计划"):
        boundary.app.generate("Show a one-second pause", dry_run=True)
    root, manifest = saved_run(boundary)
    assert manifest.status == "failed"
    assert manifest.request_budget.run_limit == 1
    assert len(manifest.request_budget.reservations) == 1
    assert boundary.calls == ["outline"]
    assert not boundary.backend.sources
    assert not list(root.rglob("*.py"))


@pytest.mark.parametrize("entry", ["retry", "incremental"])
def test_legacy_manifest_is_read_only_at_execution_entries(application_boundary, entry):
    boundary = application_boundary
    boundary.app.plan_only("Show a one-second pause")
    root, manifest = saved_run(boundary)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 9
    original = json.dumps(payload)
    manifest_path.write_text(original, encoding="utf-8")
    calls = list(boundary.calls)
    with pytest.raises(ValueError, match="只读"):
        if entry == "retry":
            boundary.app.retry(manifest.run_id, 1)
        else:
            boundary.app.run_incremental("Show a different pause", manifest.run_id)
    assert manifest_path.read_text(encoding="utf-8") == original
    assert boundary.calls == calls
    assert not boundary.backend.sources
    assert len(list((boundary.workspace / "runs").glob("*/manifest.json"))) == 1


def test_incremental_reuses_verified_artifact_without_mutating_base_or_request_budget(
    application_boundary,
):
    boundary = application_boundary
    boundary.app.generate("Show a one-second pause")
    base_root, base = saved_run(boundary)
    base_bytes = (base_root / "manifest.json").read_bytes()
    base_requests = {item.request_id for item in base.request_budget.reservations}

    video = boundary.app.run_incremental("Show a one-second pause", base.run_id)

    assert video is not None and video.read_bytes() == b"merged:" + VIDEO
    roots = [path for path in (boundary.workspace / "runs").iterdir() if path != base_root]
    assert len(roots) == 1
    incremental = RunRepository(boundary.workspace).load_for_resume(roots[0].name)
    assert incremental.status == "completed" and incremental.base_run_id == base.run_id
    assert incremental.scenes[1].artifact.origin == "reused"
    assert incremental.scenes[1].artifact.video_sha256 == base.scenes[1].artifact.video_sha256
    assert len(boundary.backend.sources) == 1
    assert (base_root / "manifest.json").read_bytes() == base_bytes
    assert incremental.request_budget.reservations
    assert base_requests.isdisjoint(
        item.request_id for item in incremental.request_budget.reservations
    )


def test_direct_render_does_not_require_cross_scene_export_definitions(
    application_boundary, monkeypatch
):
    boundary = application_boundary
    source = (
        "from manim import *\nclass RadiusScene(Scene):\n"
        "    def construct(self):\n"
        "        radius = 1\n"
        "        circle = Circle(radius=radius)\n"
        "        self.add(circle)\n"
        "        self.wait(1)\n"
    )
    monkeypatch.setattr("test_application.SOURCE", source)
    job, video, run_id = boundary.app.submit_existing_scene(source, "RadiusScene", wait=True)
    assert video.read_bytes() == b"merged:" + VIDEO
    manifest = RunRepository(boundary.workspace).load_for_resume(run_id)
    assert manifest.direct_render and manifest.status == "completed"
    assert manifest.scenes[1].code_sha256 == sha256_text(source)
    assert job.job_id == "1" and len(boundary.backend.sources) == 1
    assert not manifest.request_budget.reservations
