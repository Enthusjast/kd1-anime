"""Typed scene commits exercise the real v11 durable manifest boundary."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from pydantic import ValidationError

import kd1_anime.commits as commits
from kd1_anime.agent_runtime.state import AgentRunState
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch, ScenePatch
from kd1_anime.domain.dependencies import (
    DependencyBinding,
    DependencyGraph,
    StaleWorkError,
    WorkIdentity,
)
from kd1_anime.domain.execution import StageAttempts
from kd1_anime.domain.planning import SceneOutline, ScenePlan
from kd1_anime.prompt_identity import capture_generation_identity
from kd1_anime.request_budget import RequestBudget, RequestBudgetExceeded, RequestReservation
from kd1_anime.run_store import RunManifest, RunRepository, StoredSceneState, write_manifest

RUN_ID = "20261027-120000-1234abcd"


def scene(scene_id):
    return StoredSceneState(
        plan=ScenePlan(
            scene_id=scene_id,
            title=f"scene {scene_id}",
            duration_seconds=10,
            purpose="test",
            math_concept="circle",
            visual_design="dark",
            camera_movement="fixed",
            visual_flow=["show"],
            key_moments=["pause"],
            computation="radius=1",
            dependencies_complete=True,
        ),
        input_sha256=str(scene_id) * 64,
    )


def durable_run(tmp_path, **changes):
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    manifest = RunManifest(
        schema_version=11,
        agent=changes.pop(
            "agent",
            AgentRunState(
                session_id="commit-session",
                generation_identity=capture_generation_identity(),
                run_input_sha256="a" * 64,
            ),
        ),
        run_id=RUN_ID,
        user_prompt="prompt",
        output_path="output.mp4",
        scenes={1: scene(1), 2: scene(2), 3: scene(3)},
        **changes,
    )
    write_manifest(root / "manifest.json", manifest)
    assert manifest.integrity_errors() == []
    return root, manifest, RunCommitter(root, manifest)


def proposal(manifest, scene_id, *, dependencies=None, **changes):
    current = manifest.scenes[scene_id]
    replacement = current.model_copy(update=changes)
    patch = ScenePatch(
        scene_id=scene_id,
        identity=WorkIdentity(current.revision, current.input_sha256),
        replacement=replacement,
        dependency_revisions=dependencies or {},
    )
    return CommitCommand(manifest.revision, "CODING", patch=RunPatch(scenes=(patch,)))


def load(tmp_path):
    saved = RunRepository(tmp_path).load(RUN_ID)
    assert saved.schema_version == 11
    assert saved.integrity_errors() == []
    return saved


def reservation(request_id, **changes):
    return RequestReservation(
        request_id=request_id,
        stage="coding",
        profile="main",
        model="test",
        **changes,
    )


def test_typed_commit_persists_before_publication_and_preserves_untouched_scenes(
    monkeypatch, tmp_path
):
    root, original, committer = durable_run(tmp_path)
    real_write = commits.write_manifest
    seen = []

    def write(path, candidate):
        assert committer.current is original
        assert original.scenes[1].phase == "pending"
        real_write(path, candidate)
        seen.append(load(tmp_path).revision)

    monkeypatch.setattr(commits, "write_manifest", write)
    command = proposal(original, 1, phase="detailed", input_sha256="a" * 64)
    published = committer.commit(command)
    assert published is committer.current
    assert published.revision == 1
    assert published.scenes[1].revision == 1
    assert published.scenes[1].input_sha256 == "a" * 64
    assert published.scenes[2] == original.scenes[2]
    assert published.scenes[3] == original.scenes[3]
    assert original.revision == original.scenes[1].revision == 0
    assert original.scenes[1].phase == "pending"
    assert seen == [1]
    assert load(tmp_path).model_dump(mode="json") == published.model_dump(mode="json")
    assert (root / "manifest.json").stat().st_mode & 0o777 == 0o600


def test_same_manifest_revision_has_one_concurrent_winner(tmp_path):
    _, original, committer = durable_run(tmp_path)
    commands = [
        proposal(original, 1, phase="detailed"),
        proposal(original, 1, phase="detailed", input_sha256="b" * 64),
    ]
    ready = Barrier(2)

    def commit(command):
        ready.wait()
        try:
            return committer.commit(command)
        except StaleWorkError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, commands))
    assert sum(result is not None for result in results) == 1
    assert committer.current.revision == load(tmp_path).revision == 1
    assert committer.current.scenes[1].revision == 1


def test_independent_scene_requires_explicit_rebase_without_repeating_work(tmp_path):
    _, original, committer = durable_run(tmp_path)
    first = proposal(original, 1, phase="detailed")
    second = proposal(original, 2, phase="detailed")
    committer.commit(first)
    with pytest.raises(StaleWorkError):
        committer.commit(second)
    rebased = committer.rebase(second)
    assert second.expected_revision == 0
    assert rebased.expected_revision == 1
    assert rebased.patch == second.patch
    assert rebased.patch.scenes[0].replacement == second.patch.scenes[0].replacement
    committer.commit(rebased)
    saved = load(tmp_path)
    assert saved.revision == 2
    assert saved.scenes[1].revision == saved.scenes[2].revision == 1
    assert saved.scenes[1].phase == saved.scenes[2].phase == "detailed"


@pytest.mark.parametrize("changes", [{"phase": "detailed"}, {"input_sha256": "b" * 64}])
def test_same_scene_stale_result_cannot_be_rebased(tmp_path, changes):
    _, original, committer = durable_run(tmp_path)
    stale = proposal(original, 1, phase="detailed")
    committer.commit(proposal(original, 1, **changes))
    with pytest.raises(StaleWorkError):
        committer.rebase(stale)
    with pytest.raises(StaleWorkError):
        committer.commit(replace(stale, expected_revision=1))
    assert load(tmp_path).revision == 1


def test_identity_digest_is_checked_even_when_revision_matches(tmp_path):
    _, original, committer = durable_run(tmp_path)
    command = proposal(original, 1, phase="detailed")
    patch = command.patch.scenes[0].model_copy(update={"identity": WorkIdentity(0, "f" * 64)})
    command = replace(command, patch=RunPatch(scenes=(patch,)))
    with pytest.raises(StaleWorkError):
        committer.commit(command)
    assert committer.current is original
    assert load(tmp_path).revision == 0


def test_dependency_revision_is_checked_after_manual_manifest_rebase(tmp_path):
    graph = DependencyGraph(
        bindings=[
            DependencyBinding(
                source_scene_id=1, consumer_scene_id=2, kind="teaching", source_sha256="a" * 64
            )
        ]
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    consumer = proposal(original, 2, dependencies={1: 0}, phase="detailed")
    committer.commit(proposal(original, 1, phase="detailed"))
    with pytest.raises(StaleWorkError):
        committer.rebase(consumer)
    with pytest.raises(StaleWorkError):
        committer.commit(replace(consumer, expected_revision=1))
    assert load(tmp_path).scenes[2].revision == 0


def test_declared_dependencies_cannot_be_omitted(tmp_path):
    graph = DependencyGraph(
        bindings=[
            DependencyBinding(
                source_scene_id=1, consumer_scene_id=2, kind="handoff", source_sha256="a" * 64
            )
        ]
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    with pytest.raises(ValueError, match="dependenc"):
        committer.commit(proposal(original, 2, phase="detailed"))
    assert load(tmp_path).revision == 0


def test_reservations_survive_rebased_scene_commits_and_duplicate_replay(tmp_path):
    _, original, committer = durable_run(tmp_path)
    delayed = proposal(original, 2, phase="detailed")
    first = reservation("request-one", scene_id=1)
    second = reservation("request-two", scene_id=2)
    committer.commit(CommitCommand(0, "CODING", patch=RunPatch(reservation_additions=(first,))))
    delayed = replace(
        delayed,
        patch=RunPatch(scenes=delayed.patch.scenes, reservation_additions=(second,)),
    )
    committer.commit(committer.rebase(delayed))
    committer.commit(CommitCommand(2, "CODING", patch=RunPatch(reservation_additions=(first,))))
    assert load(tmp_path).request_budget.reservations == [first, second]
    assert original.request_budget.reservations == []


def test_conflicting_reservation_id_rejected_without_publication(tmp_path):
    _, _, committer = durable_run(tmp_path)
    first = reservation("duplicate", scene_id=1)
    committer.commit(CommitCommand(0, "CODING", patch=RunPatch(reservation_additions=(first,))))
    current = committer.current
    conflicting = first.model_copy(update={"model": "different"})
    with pytest.raises(ValueError, match="reservation"):
        committer.commit(
            CommitCommand(1, "CODING", patch=RunPatch(reservation_additions=(conflicting,)))
        )
    assert committer.current is current
    assert load(tmp_path).request_budget.reservations == [first]


def test_reservation_additions_obey_original_limits(tmp_path):
    _, original, committer = durable_run(tmp_path, request_budget=RequestBudget(run_limit=1))
    with pytest.raises(RequestBudgetExceeded):
        committer.commit(
            CommitCommand(
                0,
                "CODING",
                patch=RunPatch(reservation_additions=(reservation("one"), reservation("two"))),
            )
        )
    assert committer.current is original
    assert load(tmp_path).request_budget.reservations == []


def test_stage_attempts_keep_maximum_for_each_operation_and_input(tmp_path):
    _, original, committer = durable_run(tmp_path)
    generation = StageAttempts(operation="generation", input_sha256="a" * 64, used=2)
    technical = StageAttempts(operation="technical", input_sha256="a" * 64, used=1)
    current = committer.commit(proposal(original, 1, stage_attempts=[generation, technical]))
    old_generation = StageAttempts(operation="generation", input_sha256="a" * 64, used=1)
    different_input = StageAttempts(operation="generation", input_sha256="b" * 64, used=1)
    committer.commit(proposal(current, 1, stage_attempts=[old_generation, different_input]))
    saved = load(tmp_path)
    assert {
        (item.operation, item.input_sha256): item.used for item in saved.scenes[1].stage_attempts
    } == {
        ("generation", "a" * 64): 2,
        ("technical", "a" * 64): 1,
        ("generation", "b" * 64): 1,
    }


def test_invalid_manifest_is_rejected_instead_of_persisted(tmp_path):
    root, original, committer = durable_run(tmp_path)
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="完整性"):
        committer.commit(proposal(original, 1, rendered=True))
    assert committer.current is original
    assert (root / "manifest.json").read_bytes() == before


def test_write_failure_does_not_publish_or_mutate_original_scenes(monkeypatch, tmp_path):
    root, original, committer = durable_run(tmp_path)
    before = original.model_dump_json()
    disk_before = (root / "manifest.json").read_bytes()
    scenes = dict(original.scenes)

    def fail_after_writer_mutation(path, text):
        raise OSError("disk full")

    monkeypatch.setattr("kd1_anime.run_store.atomic_write_text", fail_after_writer_mutation)
    with pytest.raises(OSError, match="disk full"):
        committer.commit(proposal(original, 1, phase="detailed"))
    assert committer.current is original
    assert original.model_dump_json() == before
    assert all(original.scenes[key] is value for key, value in scenes.items())
    assert (root / "manifest.json").read_bytes() == disk_before


def test_patch_models_are_closed_frozen_and_disallow_unbounded_mutation():
    with pytest.raises(ValidationError):
        RunPatch(request_budget=RequestBudget())
    with pytest.raises(ValidationError):
        RunPatch(user_prompt="overwrite pinned input")
    with pytest.raises(ValidationError):
        RunPatch(state="UNKNOWN")
    with pytest.raises(ValidationError):
        RunPatch(error=None)
    patch = RunPatch(error="failed")
    with pytest.raises(ValidationError):
        patch.error = "mutated"
    with pytest.raises(ValidationError):
        ScenePatch(scene_id=1, identity=WorkIdentity(0, "a" * 64), replacement=scene(2))


def test_unset_run_fields_are_preserved_and_explicit_nullable_field_clears(tmp_path):
    _, original, committer = durable_run(tmp_path, error="keep", expected_final_duration=15)
    current = committer.commit(
        CommitCommand(0, "CODING", patch=RunPatch(expected_final_duration=None))
    )
    assert current.error == "keep"
    assert current.expected_final_duration is None
    assert load(tmp_path).expected_final_duration is None
    assert original.expected_final_duration == 15


def test_run_field_patch_cannot_be_blindly_rebased(tmp_path):
    _, original, committer = durable_run(tmp_path)
    stale = CommitCommand(0, "CODING", patch=RunPatch(error="stale error"))
    committer.commit(proposal(original, 1, phase="detailed"))
    with pytest.raises(StaleWorkError):
        committer.rebase(stale)
    assert load(tmp_path).error == ""


def test_duplicate_scene_patches_rejected():
    item = ScenePatch(scene_id=1, identity=WorkIdentity(0, "1" * 64), replacement=scene(1))
    with pytest.raises(ValidationError):
        RunPatch(scenes=(item, item))


def test_legacy_apply_is_rejected_by_command_constructor(tmp_path):
    _, original, committer = durable_run(tmp_path)
    with pytest.raises(TypeError, match="apply"):
        CommitCommand(0, "CODING", apply=lambda draft: None)
    assert committer.current is original


def test_command_identity_cannot_disagree_with_typed_scene_patch(tmp_path):
    _, original, committer = durable_run(tmp_path)
    command = proposal(original, 1, phase="detailed")
    command = replace(command, scene_id=1, identity=WorkIdentity(0, "f" * 64))
    with pytest.raises(StaleWorkError):
        committer.commit(command)
    assert load(tmp_path).revision == 0


def test_changed_upstream_teaching_requires_transitive_consumer_patches(tmp_path):
    from kd1_anime.services.dependencies import DependencyService

    graph = DependencyGraph(
        bindings=[
            DependencyBinding(
                source_scene_id=source,
                consumer_scene_id=source + 1,
                kind="teaching",
                source_sha256=DependencyService.teaching(scene(source).plan),
            )
            for source in (1, 2)
        ]
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    changed_plan = original.scenes[1].plan.model_copy(update={"computation": "radius=2"})
    source = proposal(original, 1, plan=changed_plan)
    with pytest.raises(ValueError, match="invalidation"):
        committer.commit(source)
    assert load(tmp_path).revision == 0
    second = proposal(original, 2, dependencies={1: 0}).patch.scenes[0]
    third = proposal(original, 3, dependencies={2: 0}).patch.scenes[0]
    command = replace(source, patch=RunPatch(scenes=(*source.patch.scenes, second, third)))
    current = committer.commit(command)
    assert {value.revision for value in current.scenes.values()} == {1}
    assert load(tmp_path).scenes[1].plan.computation == "radius=2"


@pytest.mark.parametrize(
    "retained",
    [
        {"exported_elements_code": "stale export"},
        {"local_smoke_status": "passed"},
        {"capability_status": "passed"},
        {"reviewed": True, "code_file": "scenes/scene_2.py"},
        {"visual_score": 4.0},
    ],
)
def test_consumer_patch_must_clear_reusable_receipts_not_just_increment_revision(
    tmp_path, retained
):
    from kd1_anime.services.dependencies import DependencyService

    graph = DependencyGraph(
        bindings=[
            DependencyBinding(
                source_scene_id=1,
                consumer_scene_id=2,
                kind="teaching",
                source_sha256=DependencyService.teaching(scene(1).plan),
            )
        ]
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    changed_plan = original.scenes[1].plan.model_copy(update={"computation": "radius=2"})
    source = proposal(original, 1, plan=changed_plan).patch.scenes[0]
    reusable = proposal(original, 2, dependencies={1: 0}, **retained).patch.scenes[0]
    third = proposal(original, 3).patch.scenes[0]
    with pytest.raises(ValueError, match="invalidation"):
        committer.commit(
            CommitCommand(0, "CODING", patch=RunPatch(scenes=(source, reusable, third)))
        )
    assert load(tmp_path).revision == 0


def test_consumer_invalidation_preserves_plan_review_but_clears_execution_outputs(tmp_path):
    from kd1_anime.services.dependencies import DependencyService

    graph = DependencyGraph(
        bindings=[
            DependencyBinding(
                source_scene_id=1,
                consumer_scene_id=2,
                kind="teaching",
                source_sha256=DependencyService.teaching(scene(1).plan),
            )
        ]
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    changed_plan = original.scenes[1].plan.model_copy(update={"computation": "radius=2"})
    source = proposal(original, 1, plan=changed_plan).patch.scenes[0]
    consumer = proposal(
        original, 2, dependencies={1: 0}, plan_ready=True, plan_reviewed=True
    ).patch.scenes[0]
    current = committer.commit(
        CommitCommand(0, "CODING", patch=RunPatch(scenes=(source, consumer)))
    )
    invalidated = current.scenes[2]
    assert invalidated.plan_ready and invalidated.plan_reviewed
    assert not invalidated.reviewed and not invalidated.rendered
    assert invalidated.technical_status == "pending"
    assert invalidated.visual_status == "skipped"
    assert invalidated.static_verification.status == "not_run"
    assert invalidated.execution_verification.status == "not_run"
    assert invalidated.visual_verification.status == "not_run"


def test_manifest_dependency_analysis_is_pure_and_ignores_revision_only_changes(tmp_path):
    from kd1_anime.services.dependencies import DependencyService

    _, original, _ = durable_run(tmp_path)
    for stored in original.scenes.values():
        stored.plan.dependencies_complete = False
    before = original.model_dump_json()
    after = original.model_copy(
        update={
            "scenes": {**original.scenes, 1: original.scenes[1].model_copy(update={"revision": 1})}
        }
    )
    assert DependencyService.affected_manifest(original, after, (1,)) == set()
    changed = after.scenes[1].plan.model_copy(update={"computation": "radius=2"})
    after.scenes[1].plan = changed
    assert DependencyService.affected_manifest(original, after, (1,)) == {2, 3}
    assert original.model_dump_json() == before


def test_global_input_change_requires_explicit_consumer_invalidation(tmp_path):
    from kd1_anime.domain.planning import LessonSpec

    graph = DependencyGraph(global_inputs={1: "a" * 64})
    _, _, committer = durable_run(tmp_path, dependency_graph=graph)
    with pytest.raises(ValueError, match="invalidation"):
        committer.commit(
            CommitCommand(0, "CODING", patch=RunPatch(lesson_spec=LessonSpec(topic="new topic")))
        )
    assert load(tmp_path).revision == 0


@pytest.mark.parametrize("revision", [True, 0.0, -1])
def test_expected_manifest_revision_is_a_strict_nonnegative_integer(revision):
    with pytest.raises(ValueError, match="revision"):
        CommitCommand(revision, "CODING", patch=RunPatch())


def test_untyped_patch_is_not_accepted_by_the_new_command():
    with pytest.raises(TypeError, match="typed RunPatch"):
        CommitCommand(0, "CODING", patch={"error": "untyped"})


def test_typed_commit_can_add_a_new_scene_without_replacing_existing_scenes(tmp_path):
    outline_identity = WorkIdentity(0, "d" * 64)
    agent = AgentRunState(
        session_id="commit-session",
        generation_identity=capture_generation_identity(),
        run_input_sha256="a" * 64,
        outline_identities={4: outline_identity},
    )
    outline = SceneOutline(
        scene_id=4,
        title="scene 4",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
    )
    _, original, committer = durable_run(tmp_path, agent=agent, outlines=[outline])
    addition = ScenePatch(
        scene_id=4,
        identity=outline_identity,
        replacement=scene(4),
    )
    current = committer.commit(CommitCommand(0, "DETAILING", patch=RunPatch(scenes=(addition,))))
    assert set(current.scenes) == {1, 2, 3, 4}
    assert current.scenes[4].revision == 1
    assert current.scenes[1] == original.scenes[1]
    assert set(load(tmp_path).scenes) == {1, 2, 3, 4}


def test_global_input_invalidation_reaches_indirect_consumers(tmp_path):
    from kd1_anime.domain.planning import LessonSpec
    from kd1_anime.services.dependencies import DependencyService

    graph = DependencyGraph(
        global_inputs={1: "a" * 64},
        bindings=[
            DependencyBinding(
                source_scene_id=source,
                consumer_scene_id=source + 1,
                kind="teaching",
                source_sha256=DependencyService.teaching(scene(source).plan),
            )
            for source in (1, 2)
        ],
    )
    _, original, committer = durable_run(tmp_path, dependency_graph=graph)
    first = proposal(original, 1).patch.scenes[0]
    with pytest.raises(ValueError, match="invalidation"):
        committer.commit(
            CommitCommand(
                0,
                "CODING",
                patch=RunPatch(scenes=(first,), lesson_spec=LessonSpec(topic="changed")),
            )
        )
    patches = (
        first,
        proposal(original, 2, dependencies={1: 0}).patch.scenes[0],
        proposal(original, 3, dependencies={2: 0}).patch.scenes[0],
    )
    committer.commit(
        CommitCommand(
            0, "CODING", patch=RunPatch(scenes=patches, lesson_spec=LessonSpec(topic="changed"))
        )
    )
    assert {stored.revision for stored in load(tmp_path).scenes.values()} == {1}
