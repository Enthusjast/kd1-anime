import pytest

from kd1_anime.agents.validator import validate_manim_code
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.exceptions import ValidationError
from kd1_anime.services.generation import GenerationService
from kd1_anime.services.planning import PlanningService
from kd1_anime.services.recovery import RecoveryInput, RecoveryService


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


def generation_plan():
    return ScenePlan(
        scene_id=1,
        title="Circle",
        duration_seconds=1,
        purpose="Show one circle",
        math_concept="Circle",
        visual_design="A circle",
        camera_movement="Fixed",
        visual_flow=["Create a circle"],
        key_moments=["Circle appears"],
        computation="radius=1",
    )


def test_generation_rejects_deprecated_api_before_accepting_revised_candidate():
    invalid = (
        "from manim import *\nclass Demo(Scene):\n"
        "    def construct(self): self.play(ShowCreation(Circle()))\n"
    )
    valid = invalid.replace("ShowCreation", "Create")

    class Coder:
        def __init__(self):
            self.calls = 0

        def generate_code(self, plan, **kwargs):
            self.calls += 1
            return invalid if self.calls == 1 else valid

    coder = Coder()
    result = GenerationService(
        agent=coder,
        validator=validate_manim_code,
        candidate_acceptor=CandidateAcceptor(),
        generation_mode="strict",
        manim_version="0.20.1",
        limits={"CODE_VALIDATION_ATTEMPTS": 3},
    ).generate(generation_plan())

    assert result.code == valid
    assert result.class_name == "Demo"
    assert result.attempts == coder.calls == 2


def test_generation_exhausts_validation_budget_without_accepting_unsafe_source():
    class Coder:
        def __init__(self):
            self.calls = 0

        def generate_code(self, plan, **kwargs):
            self.calls += 1
            return "import os\nos.system('unsafe')\n"

    coder = Coder()
    service = GenerationService(
        agent=coder,
        validator=validate_manim_code,
        candidate_acceptor=CandidateAcceptor(),
        generation_mode="strict",
        limits={"CODE_VALIDATION_ATTEMPTS": 3},
    )

    with pytest.raises(ValidationError):
        service.generate(generation_plan())
    assert coder.calls == 3
