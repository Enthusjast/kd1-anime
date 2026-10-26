from pathlib import Path

from kd1_anime.services.planning import PlanningService
from kd1_anime.services.recovery import RecoveryInput, RecoveryService
from kd1_anime.services.visual_evaluation import VisualEvaluationService


def test_planning_service_signature_is_stable_and_duration_is_bounded():
    payload = {"b": 2, "a": 1}
    assert PlanningService().cycle_signature(payload) == PlanningService().cycle_signature(
        {"a": 1, "b": 2}
    )
    assert PlanningService.expected_duration([10, 10], 0.5) == 19.5
    assert PlanningService.expected_duration([0.2, 10], 0.5) == 10.1


def test_recovery_service_returns_decisions_without_mutating_inputs():
    scenes = [RecoveryInput(1, "local-abcdef123456", False), RecoveryInput(2, "local-done", True)]
    result = RecoveryService.detach_unresumable_local_jobs(scenes)
    assert len(result) == 1
    assert result[0].scene_id == 1
    assert result[0].job_id == scenes[0].job_id
    assert scenes[0].job_id == "local-abcdef123456"


def test_visual_service_forwards_only_explicit_evaluation_inputs():
    class Evaluator:
        def evaluate_video_frames(self, samples, description, *, scene_context, scope):
            return samples, description, scene_context, scope

    result = VisualEvaluationService.evaluate_frames(
        Evaluator(),
        [Path("frame.jpg")],
        "description",
        scene_context="context",
        scope="scene",
    )
    assert result == ([Path("frame.jpg")], "description", "context", "scene")
