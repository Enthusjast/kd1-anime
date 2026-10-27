import json

from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime


def test_unknown_submission_stops_durably_and_resume_never_redispatches(
    agent_boundary, monkeypatch
):
    boundary, dependencies = agent_boundary
    submit = boundary.backend.submit_scene

    def interrupted_submission(*args, **kwargs):
        submit(*args, **kwargs)
        raise TimeoutError("provider transport state is unknown")

    monkeypatch.setattr(boundary.backend, "submit_scene", interrupted_submission)
    assert (
        AgentRuntime(dependencies=dependencies).generate(
            "Show a one-second pause", approve_plan=True
        )
        is None
    )
    root = next((boundary.workspace / "runs").iterdir())
    raw = json.loads((root / "manifest.json").read_text())
    assert raw["agent"]["stop_reason"]
    assert raw["agent"]["invocations"][-1]["status"] == "unknown"
    assert len(boundary.backend.sources) == 1
    assert AgentRuntime(dependencies=dependencies).resume(root.name, approve_plan=True) is None
    assert len(boundary.backend.sources) == 1


def test_plan_only_can_skip_probe_and_model_review_without_skipping_validation(agent_boundary):
    boundary, dependencies = agent_boundary

    def forbidden_probe(agent):
        raise AssertionError("preflight=False must not probe a model endpoint")

    dependencies.probe = forbidden_probe
    plans = AgentRuntime(dependencies=dependencies).plan_only(
        "Show a one-second pause", preflight=False, review=False
    )
    assert len(plans) == 1
    assert not boundary.backend.sources
    root = next((boundary.workspace / "runs").iterdir())
    raw = json.loads((root / "manifest.json").read_text())
    assert raw["state"] == "DONE"
    assert raw["agent"]["planning_preflight"] is False
    assert raw["plan_review_status"] == "passed"
    assert not any(
        item["stage"] in {"plan_review", "continuity_review"}
        for item in raw["request_budget"]["reservations"]
    )


def test_unsupported_rag_stops_before_any_model_or_render(agent_boundary, monkeypatch):
    from kd1_anime.config import settings

    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    assert (
        AgentRuntime(dependencies=dependencies).generate(
            "Show a one-second pause", approve_plan=True
        )
        is None
    )
    root = next((boundary.workspace / "runs").iterdir())
    raw = json.loads((root / "manifest.json").read_text())
    assert raw["status"] == "failed" and "RAG" in raw["agent"]["stop_reason"]
    assert not raw["request_budget"]["reservations"] and not boundary.backend.sources


def test_resume_rejects_plan_changed_without_identity_update(agent_boundary):
    import pytest

    from kd1_anime.run_store import RunRepository, write_manifest

    boundary, dependencies = agent_boundary
    assert AgentRuntime(dependencies=dependencies).generate("Show a one-second pause") is None
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    manifest.scenes[1].plan.title = "Unapproved replacement"
    write_manifest(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="semantic identity"):
        AgentRuntime(dependencies=dependencies).resume(root.name, approve_plan=True)
    assert not boundary.backend.sources


def test_concurrent_scene_revision_rejects_older_codegen_result(agent_boundary):
    from kd1_anime.commits import CommitCommand, RunPatch, ScenePatch
    from kd1_anime.domain.dependencies import WorkIdentity

    boundary, dependencies = agent_boundary
    original_factory = dependencies.coder_factory

    class RacingCoder(original_factory):
        def generate_code(self, *args, **kwargs):
            source = super().generate_code(*args, **kwargs)
            current = runtime.committer.current
            scene = current.scenes[1]
            replacement = scene.model_copy(deep=True)
            replacement.failure_reason = "Concurrent scene update"
            replacement.snapshot = None
            runtime.committer.commit(
                CommitCommand(
                    current.revision,
                    "CODING",
                    patch=RunPatch(
                        scenes=(
                            ScenePatch(
                                scene_id=1,
                                identity=WorkIdentity(scene.revision, scene.input_sha256),
                                replacement=replacement,
                            ),
                        )
                    ),
                )
            )
            return source

    dependencies.coder_factory = RacingCoder
    runtime = AgentRuntime(dependencies=dependencies)
    assert runtime.generate("Show a one-second pause", approve_plan=True) is None
    current = runtime.committer.current
    assert current.scenes[1].failure_reason == "Concurrent scene update"
    assert current.scenes[1].code_file == ""
    assert not boundary.backend.sources
