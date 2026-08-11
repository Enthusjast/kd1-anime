"""增量复用必须同时匹配代码、渲染配置和视频内容。"""

from pathlib import Path

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.reviewer import ReviewResult
from kd1_anime.config import settings
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState
from kd1_anime.rendering import RenderProfile, SceneArtifact, VideoMetadata, sha256_file
from kd1_anime.run_store import RunManifest, StoredSceneState, get_reusable_video_path, sha256_text

RUN_ID = "20260728-120000-1234abcd"
NEW_RUN_ID = "20260729-120000-deadbeef"
CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"


def make_plan(scene_id: int = 1) -> ScenePlan:
    return ScenePlan(
        scene_id=scene_id,
        title=f"Scene {scene_id}",
        duration_seconds=30,
        purpose="Test",
        math_concept="Test",
        visual_design="Test",
        camera_movement="Test",
        visual_flow=["Step 1"],
        key_moments=["Moment 1"],
        computation="Test",
    )


def make_paths(workspace: Path) -> RunPaths:
    root = workspace / "runs" / NEW_RUN_ID
    for directory in (root, root / "scenes", root / "logs", root / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        NEW_RUN_ID,
        root,
        root / "scenes",
        root / "logs",
        root / "videos",
        root / "output.mp4",
    )


def make_base(workspace: Path, profile: RenderProfile) -> tuple[RunManifest, Path]:
    root = workspace / "runs" / RUN_ID
    video = root / "videos" / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"verified old video")
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=RUN_ID,
        job_id="12345",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(CODE),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(root).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    manifest = RunManifest(
        run_id=RUN_ID,
        status="completed",
        state="DONE",
        user_prompt="base",
        output_path=str(root / "output.mp4"),
        render_profile=profile,
        scenes={
            1: StoredSceneState(
                plan=make_plan(),
                code_file="scenes/scene_1.py",
                code_sha256=sha256_text(CODE),
                class_name="Demo",
                reviewed=True,
                rendered=True,
                artifact=artifact,
            )
        },
    )
    return manifest, video


def test_incremental_reuses_verified_matching_artifact(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    profile = RenderProfile.current()
    base, video = make_base(workspace, profile)
    paths = make_paths(workspace)
    state = SceneState(plan=make_plan(), code=CODE, class_name="Demo", reviewed=True)
    ctx = PipelineContext(
        "updated",
        paths=paths,
        scene_states={1: state},
        incremental=True,
        base_run_id=RUN_ID,
        base_manifest=base,
        render_profile=profile,
    )

    Orchestrator()._apply_incremental_for_scene(ctx, 1, state)

    assert state.rendered is True
    assert state.slurm_job is None
    assert state.artifact is not None
    assert state.artifact.origin == "reused"
    assert get_reusable_video_path(base, 1, workspace / "runs" / RUN_ID) == video


def test_incremental_rejects_render_profile_change(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    old_profile = RenderProfile.current()
    base, _ = make_base(workspace, old_profile)
    changed_profile = old_profile.model_copy(update={"frame_rate": old_profile.frame_rate + 1})
    paths = make_paths(workspace)
    state = SceneState(plan=make_plan(), code=CODE, class_name="Demo", reviewed=True)
    ctx = PipelineContext(
        "updated",
        paths=paths,
        scene_states={1: state},
        incremental=True,
        base_run_id=RUN_ID,
        base_manifest=base,
        render_profile=changed_profile,
    )

    Orchestrator()._apply_incremental_for_scene(ctx, 1, state)

    assert state.rendered is False
    assert state.artifact is None
    assert ctx.scenes_to_render == [1]


def test_incremental_rejects_code_change(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    profile = RenderProfile.current()
    base, _ = make_base(workspace, profile)
    paths = make_paths(workspace)
    changed = CODE.replace("self.wait()", "self.wait(2)")
    state = SceneState(plan=make_plan(), code=changed, class_name="Demo", reviewed=True)
    ctx = PipelineContext(
        "updated",
        paths=paths,
        scene_states={1: state},
        incremental=True,
        base_run_id=RUN_ID,
        base_manifest=base,
        render_profile=profile,
    )

    Orchestrator()._apply_incremental_for_scene(ctx, 1, state)

    assert state.rendered is False
    assert ctx.scenes_to_render == [1]


def test_reviewer_code_change_invalidates_existing_artifact(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    profile = RenderProfile.current()
    base, _ = make_base(workspace, profile)
    paths = make_paths(workspace)
    paths.scenes.joinpath("scene_1.py").write_text(CODE, encoding="utf-8")
    artifact = base.scenes[1].artifact
    state = SceneState(
        plan=make_plan(),
        code=CODE,
        class_name="Demo",
        artifact=artifact,
        rendered=True,
    )
    ctx = PipelineContext("x", paths=paths, scene_states={1: state}, render_profile=profile)
    orchestrator = Orchestrator()

    orchestrator._apply_review_result(
        ctx,
        1,
        state,
        ReviewResult(
            is_valid=False,
            severity="minor",
            feedback="增加停顿",
            fixes=[
                {
                    "find": "self.wait()",
                    "replace": "self.wait(2)",
                    "reason": "节奏",
                }
            ],
        ),
    )

    assert state.code != CODE
    assert state.rendered is False
    assert state.artifact is None
