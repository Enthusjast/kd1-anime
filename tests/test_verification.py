from pathlib import Path

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.run_store import RunManifest, RunRepository, StoredSceneState, sha256_text
from kd1_anime.verification import ExecutionVerification, StaticVerification, VisualVerification


def make_plan() -> ScenePlan:
    return ScenePlan(
        scene_id=1,
        title="测试",
        duration_seconds=1,
        purpose="验证收据",
        math_concept="等待",
        visual_design="简洁",
        camera_movement="固定",
        visual_flow=["等待"],
        key_moments=["等待"],
        computation="无",
    )


def test_static_execution_visual_receipts_are_independent():
    assert StaticVerification().status == "not_run"
    assert ExecutionVerification().status == "not_run"
    assert VisualVerification().status == "not_run"


def test_checkpoint_persists_three_verification_conclusions(tmp_path: Path):
    workspace = tmp_path / "workspace"
    root = workspace / "runs" / "20260905-120000-1234abcd"
    paths = RunPaths(
        "20260905-120000-1234abcd",
        root,
        root / "scenes",
        root / "logs",
        root / "videos",
        root / "output.mp4",
    )
    paths.scenes.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    (paths.scenes / "scene_1.py").write_text(code, encoding="utf-8")
    state = SceneState(plan=make_plan(), code=code, class_name="Demo", reviewed=True)
    orchestrator = Orchestrator()
    context = PipelineContext(
        "prompt",
        paths=paths,
        scene_states={1: state},
    )
    orchestrator._mark_static_verification(state, status="passed")
    orchestrator._mark_execution_verification(state, status="unknown", scope="formal_video")
    state.visual_verification = VisualVerification(status="unknown")
    orchestrator._checkpoint(context, State.REVIEWING)

    manifest = RunRepository(workspace).load(paths.run_id)
    stored = manifest.scenes[1]
    assert stored.static_verification.status == "passed"
    assert stored.execution_verification.status == "unknown"
    assert stored.visual_verification.status == "unknown"


def test_integrity_catches_stale_static_receipt():
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    scene = StoredSceneState(
        plan=make_plan(),
        code_sha256=sha256_text(code),
        class_name="Demo",
        static_verification=StaticVerification(status="passed", code_sha256="a" * 64),
    )
    manifest = RunManifest(
        run_id="20260905-120000-1234abcd",
        user_prompt="prompt",
        output_path="/tmp/output.mp4",
        scenes={1: scene},
    )

    assert any("静态验证代码哈希不一致" in error for error in manifest.integrity_errors())
