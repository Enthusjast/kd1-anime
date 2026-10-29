import pytest
from test_pipeline_scene import inputs

from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.domain.dependencies import DependencyBinding, digest
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.rag.models import RagRuntimeProfile
from kd1_anime.run_store import RunManifest
from kd1_anime.services.dependencies import DependencyService


@pytest.mark.parametrize(
    "changed",
    [
        {"index_sha256": "b" * 64},
        {"embedding_model": "other-embedding"},
        {"reranker_model": "other-reranker"},
        {"chunker_version": "other-chunker"},
    ],
)
def test_fixed_rag_changes_invalidate_global_consumers_and_transitive_scenes(tmp_path, changed):
    scene = inputs(tmp_path).scene
    before = RunManifest(
        agent=AgentRunState(
            session_id="session-one",
            generation_identity=capture_generation_identity(),
            run_input_sha256="a" * 64,
        ),
        run_id="20260101-000000-1234abcd",
        user_prompt="Circle",
        output_path="output.mp4",
        scenes={1: scene},
        rag_profile=RagRuntimeProfile(
            enabled=True,
            status="active",
            index_sha256="a" * 64,
            embedding_model="embedding",
            reranker_model="reranker",
            chunker_version="chunker",
        ),
    )
    global_hash = digest(
        {
            "lesson": before.lesson_spec.model_dump(mode="json"),
            "teaching": before.teaching_graph.model_dump(mode="json"),
            "bible": None,
            "renderer": before.render_profile.digest(),
            "rag": before.rag_profile.model_dump(mode="json"),
        }
    )
    before.dependency_graph.global_inputs = {1: global_hash}
    before.dependency_graph.bindings = [
        DependencyBinding(
            source_scene_id=1,
            consumer_scene_id=2,
            kind="handoff",
            source_sha256="c" * 64,
        )
    ]
    after = before.model_copy(deep=True)
    after.rag_profile = before.rag_profile.model_copy(update=changed)
    assert DependencyService.affected_manifest(before, after, ()) == {1, 2}
    assert DependencyService.affected_manifest(before, before.model_copy(deep=True), ()) == set()
    assert before.dependency_graph.global_inputs == {1: global_hash}
