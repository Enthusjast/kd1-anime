from kd1_anime.agent_runtime.contracts import ACTION_ADAPTER
from kd1_anime.agent_runtime.state import AgentRunState, merge_transport_state
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.prompt_identity import capture_generation_identity


def test_detail_scene_is_a_closed_identity_bound_action():
    identity = WorkIdentity(0, "a" * 64)
    action = ACTION_ADAPTER.validate_python(
        {
            "kind": "detail_scene",
            "run_id": "20260101-120000-12345678",
            "resource_id": "run:20260101-120000-12345678/scene:2",
            "scene_id": 2,
            "input_sha256": identity.input_sha256,
            "work_identity": identity,
        }
    )
    assert action.kind == "detail_scene" and action.work_identity == identity


def test_transport_merge_preserves_independent_scene_progress():
    base = AgentRunState(
        session_id="session",
        generation_identity=capture_generation_identity(),
        run_input_sha256="a" * 64,
        scene_next_actions={1: "propose_code_candidate", 2: "propose_code_candidate"},
    )
    first = base.model_copy(
        update={
            "scene_next_actions": {1: "inspect_candidate", 2: "propose_code_candidate"},
            "scene_action_versions": {1: 1},
        }
    )
    second = base.model_copy(
        update={
            "scene_next_actions": {1: "propose_code_candidate", 2: "inspect_candidate"},
            "scene_action_versions": {2: 1},
        }
    )
    merged = merge_transport_state(first, second)
    assert merged.scene_next_actions == {1: "inspect_candidate", 2: "inspect_candidate"}
