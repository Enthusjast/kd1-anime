from pathlib import Path
from types import SimpleNamespace

from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_pipeline_scene_candidates import fix_result

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.config import settings
from kd1_anime.run_store import RunRepository, restore_run_path


def test_automatic_review_fix_requires_fresh_smoke_for_exact_new_source(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "LOCAL_SMOKE_RENDER_MODE", "both")
    monkeypatch.setattr(settings, "ADAPTIVE_SMOKE_RENDER", False)
    reviewer = dependencies.reviewer_factory
    sources = []

    class Reviewer(reviewer):
        def review(self, code, plan, **kwargs):
            accepted = super().review(code, plan, **kwargs)
            if "self.wait(1)" in code:
                return fix_result(minor=True, deterministic=True)
            return accepted

    def execute(command, **kwargs):
        if "--media_dir" in command:
            directory = Path(command[command.index("--media_dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            source = Path(command[-2])
            sources.append(source.read_text())
            (
                directory / ("Pause.png" if "--save_last_frame" in command else "Pause.mp4")
            ).write_bytes(b"isolated-media")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    dependencies.reviewer_factory = Reviewer
    monkeypatch.setattr("kd1_anime.pipeline.smoke._run_limited_process", execute)
    runtime = AgentRuntime(dependencies=dependencies)
    assert runtime.generate("Show a pause", approve_plan=True, dry_run=True, smoke=True) is None
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    scene = manifest.scenes[1]
    source = restore_run_path(runtime.root, scene.code_file).read_text()
    assert manifest.status == "dry_run_complete" and "self.wait(2)" in source
    assert any("self.wait(1)" in item for item in sources) and sources[-1] == source
    assert scene.local_smoke_status == "passed" and scene.execution_verification.status == "passed"
    assert scene.execution_verification.code_sha256 == scene.code_sha256
    assert scene.reviewed and not boundary.backend.sources
