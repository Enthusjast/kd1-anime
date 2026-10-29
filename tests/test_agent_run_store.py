import json

import pytest

from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.run_store import RunManifest, RunRepository, write_manifest

RUN = "20261027-120000-1234abcd"


def agent_manifest():
    return RunManifest(
        schema_version=11,
        run_id=RUN,
        user_prompt="Circle",
        output_path="output.mp4",
        agent=AgentRunState(
            session_id="session-one",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
    )


def test_v11_roundtrip_requires_identity_and_never_serializes_boolean_approval(tmp_path):
    manifest = agent_manifest()
    root = tmp_path / "runs" / RUN
    write_manifest(root / "manifest.json", manifest)
    raw = json.loads((root / "manifest.json").read_text())
    assert "plan_approved" not in raw and "approve_plan" not in raw
    loaded = RunRepository(tmp_path).load_for_resume(RUN)
    assert loaded.agent == manifest.agent
    assert loaded.model_dump(mode="json") == raw
    for invalid in ({**raw, "agent": None}, {**raw, "plan_approved": True}, {**raw, "agent": {}}):
        with pytest.raises(ValueError):
            RunManifest.model_validate(invalid)


def test_v10_manifest_is_inspectable_but_not_executable_or_writable(tmp_path):
    root = tmp_path / "runs" / RUN
    write_manifest(root / "manifest.json", agent_manifest())
    raw = json.loads((root / "manifest.json").read_text())
    raw["schema_version"] = 10
    raw.pop("agent")
    raw["plan_approved"] = False
    (root / "manifest.json").write_text(json.dumps(raw))
    before = (root / "manifest.json").read_bytes()
    repository = RunRepository(tmp_path)
    loaded = repository.load(RUN)
    assert loaded.schema_version == 10
    with pytest.raises(ValueError):
        repository.load_for_resume(RUN)
    with pytest.raises(ValueError):
        write_manifest(root / "manifest.json", loaded)
    assert (root / "manifest.json").read_bytes() == before


def test_v11_typed_transport_updates_preserve_pending_intent_and_budget(tmp_path):
    from kd1_anime.agent_runtime.contracts import ExecutionIntent, ProposePlan
    from kd1_anime.domain.dependencies import WorkIdentity
    from kd1_anime.request_budget import RequestReservation

    manifest = agent_manifest()
    root = tmp_path / "runs" / RUN
    write_manifest(root / "manifest.json", manifest)
    committer = RunCommitter(root, manifest)
    action = ProposePlan(
        run_id=RUN,
        resource_id=f"run:{RUN}",
        input_sha256="a" * 64,
        work_identity=WorkIdentity(0, "a" * 64),
    )
    intent = ExecutionIntent(session_id="session-one", action=action)
    updated = manifest.agent.model_copy(update={"intents": (intent,)})
    committer.commit(CommitCommand(0, "PLANNING", patch=RunPatch(agent=updated)))
    reservation = RequestReservation(stage="planning", profile="main", model="fake")
    committer.commit(
        CommitCommand(1, "PLANNING", patch=RunPatch(reservation_additions=(reservation,)))
    )
    loaded = RunRepository(tmp_path).load_for_resume(RUN)
    assert loaded.agent.intents == (intent,)
    assert loaded.request_budget.reservations == [reservation]
    # A stage draft predating a transport reservation may advance its stage,
    # but cannot erase the request/intent that was durably reserved meanwhile.
    stale_agent = manifest.agent.model_copy(update={"next_action": "review_plan"})
    committer.commit(CommitCommand(2, "PLAN_REVIEWING", patch=RunPatch(agent=stale_agent)))
    latest = RunRepository(tmp_path).load_for_resume(RUN)
    assert latest.agent.intents == (intent,)
    assert latest.request_budget.reservations == [reservation]
