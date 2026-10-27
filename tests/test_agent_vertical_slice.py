import json

import pytest
from test_application import SOURCE, VIDEO, application_boundary

import kd1_anime.orchestrator as legacy
from kd1_anime.agent_runtime.runtime import AgentDependencies, AgentRuntime
from kd1_anime.domain.planning import LessonSpec, PlanningDraft, TeachingGraph
from kd1_anime.run_store import RunRepository


@pytest.fixture
def agent_boundary(monkeypatch, tmp_path):
    boundary = application_boundary.__wrapped__(monkeypatch, tmp_path)

    class DraftPlanner(legacy.PlannerAgent):
        def plan_draft(self, prompt, **kwargs):
            return PlanningDraft(
                items=self.plan_outline(prompt, **kwargs),
                lesson_spec=LessonSpec(topic=prompt),
                teaching_graph=TeachingGraph(),
            )

    dependencies = AgentDependencies(
        planner_factory=DraftPlanner,
        plan_reviewer_factory=legacy.PlanReviewerAgent,
        continuity_reviewer_factory=legacy.ContinuityReviewerAgent,
        technical_factory=legacy.TechnicalPlannerAgent,
        coder_factory=legacy.CoderAgent,
        reviewer_factory=legacy.ReviewerAgent,
        backend_factory=lambda name: boundary.backend,
        merger_factory=legacy.VideoMerger,
        preflight=lambda profile, backend: None,
        probe=lambda agent: None,
    )
    return boundary, dependencies


def test_single_scene_v11_uses_intents_policy_typed_commits_and_real_objects(agent_boundary):
    boundary, dependencies = agent_boundary
    output = AgentRuntime(dependencies=dependencies).generate(
        "Show a one-second pause", approve_plan=True
    )
    assert output.read_bytes() == b"merged:" + VIDEO
    roots = list((boundary.workspace / "runs").iterdir())
    assert len(roots) == 1
    raw = json.loads((roots[0] / "manifest.json").read_text())
    assert raw["schema_version"] == 11 and raw["agent"]["runtime"] == "agent"
    assert "plan_approved" not in raw
    assert raw["agent"]["session_id"]
    assert raw["agent"]["generation_identity"]["prompt_pack_sha256"]
    assert raw["agent"]["intents"] and raw["agent"]["invocations"]
    assert raw["request_budget"]["reservations"]
    source = roots[0] / raw["scenes"]["1"]["code_file"]
    assert source.read_text() == SOURCE
    assert raw["status"] == "completed"
    assert AgentRuntime(dependencies=dependencies).resume(roots[0].name) == output
    assert len(boundary.backend.sources) == 1


def test_noninteractive_plan_approval_wait_is_durable_and_never_submits(agent_boundary):
    boundary, dependencies = agent_boundary
    result = AgentRuntime(dependencies=dependencies).generate("Show a one-second pause")
    assert result is None
    root = next((boundary.workspace / "runs").iterdir())
    raw = json.loads((root / "manifest.json").read_text())
    assert raw["agent"]["pending_approval"] is not None
    assert not boundary.backend.sources
    before = len(raw["request_budget"]["reservations"])
    assert AgentRuntime(dependencies=dependencies).resume(root.name) is None
    after = json.loads((root / "manifest.json").read_text())
    assert len(after["request_budget"]["reservations"]) == before


@pytest.mark.parametrize(
    "next_action",
    [
        "detail_scene",
        "review_plan",
        "propose_technical_spec",
        "propose_code_candidate",
        "inspect_candidate",
        "review_code",
        "submit_render",
        "poll_render",
        "merge_video",
        "evaluate_run",
        "finish",
    ],
)
def test_each_committed_single_scene_stage_can_resume_without_repeating_render(
    agent_boundary, next_action
):
    boundary, dependencies = agent_boundary
    interrupted = []

    def stop_after_commit(manifest):
        pending = {manifest.agent.next_action, *manifest.agent.scene_next_actions.values()}
        if next_action in pending and not interrupted:
            interrupted.append(True)
            raise OSError("isolated stage interruption")

    dependencies.after_commit = stop_after_commit
    with pytest.raises(OSError):
        AgentRuntime(dependencies=dependencies).generate(
            "Show a one-second pause", approve_plan=True
        )
    dependencies.after_commit = None
    root = next((boundary.workspace / "runs").iterdir())
    output = AgentRuntime(dependencies=dependencies).resume(root.name, approve_plan=True)
    assert output.read_bytes() == b"merged:" + VIDEO
    assert len(boundary.backend.sources) == 1
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.status == "completed" and manifest.schema_version == 11


def test_cli_reports_approval_wait_without_claiming_video_success(agent_boundary, monkeypatch):
    from typer.testing import CliRunner

    from kd1_anime.cli import app

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    monkeypatch.setattr("kd1_anime.cli._ensure_llm_api_available", lambda: None)
    monkeypatch.setattr("kd1_anime.cli._ensure_rag_apis_available", lambda: None)
    result = CliRunner().invoke(app, ["generate", "Show a one-second pause"])
    assert result.exit_code == 0, result.output
    assert "批准" in result.output and "成功!" not in result.output
    assert not boundary.backend.sources
    root = next((boundary.workspace / "runs").iterdir())
    before = json.loads((root / "manifest.json").read_text())["request_budget"]
    resumed = CliRunner().invoke(app, ["resume", root.name])
    assert resumed.exit_code == 0, resumed.output
    assert "批准" in resumed.output and "运行已完成" not in resumed.output
    assert json.loads((root / "manifest.json").read_text())["request_budget"] == before


def test_callback_errors_do_not_change_execution_or_expose_configured_credentials(agent_boundary):
    boundary, dependencies = agent_boundary

    def failed_display(name, data):
        raise RuntimeError("display disconnected")

    output = AgentRuntime(dependencies=dependencies).generate(
        "Show a one-second pause application-test-key",
        approve_plan=True,
        callback=failed_display,
    )
    assert output.read_bytes() == b"merged:" + VIDEO
    for path in boundary.workspace.rglob("*.json"):
        assert "application-test-key" not in path.read_text()
    for path in boundary.workspace.rglob("*.jsonl"):
        assert "application-test-key" not in path.read_text()
    assert len(boundary.backend.sources) == 1


def test_cli_output_option_publishes_requested_path_on_agent_runtime(agent_boundary, monkeypatch):
    from typer.testing import CliRunner

    from kd1_anime.cli import app

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    requested = boundary.workspace / "requested.mp4"
    result = CliRunner().invoke(
        app, ["generate", "Show a one-second pause", "--approve-plan", "--output", str(requested)]
    )
    assert result.exit_code == 0, result.output
    assert requested.read_bytes() == b"merged:" + VIDEO
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.output_path == str(requested)
    assert manifest.final_video == str(requested)


def test_cli_compiler_fallback_is_explicit_and_keeps_agent_execution(agent_boundary, monkeypatch):
    from typer.testing import CliRunner

    from kd1_anime.cli import app
    from kd1_anime.config import settings

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "SCENE_CODEGEN_MODE", "compiler_first")
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    result = CliRunner().invoke(app, ["generate", "Show a one-second pause", "--approve-plan"])
    assert result.exit_code == 0, result.output
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.schema_version == 11 and manifest.status == "completed"
    generated = next(
        item for item in manifest.agent.invocations if item.action == "propose_code_candidate"
    )
    report = json.loads((root / generated.result_ref.path).read_text())
    assert any(item["code"] == "scene_program_unsupported" for item in report["diagnostics"])
    assert "code" in boundary.calls and len(boundary.backend.sources) == 1
