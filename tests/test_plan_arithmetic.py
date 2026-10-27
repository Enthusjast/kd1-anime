import pytest

from kd1_anime.agents.plan_compiler import PlanCompiler
from kd1_anime.domain.planning import ContinuityBible, SceneOutline, ScenePlan


@pytest.mark.parametrize(
    "computation,valid",
    [
        ("1+1=3", False),
        ("1+1=2", True),
        ("2*(3-1)=4", True),
        ("1=2", False),
        ("radius=1", True),
        ("1→2→3", True),
    ],
)
def test_explicit_numeric_computation_equalities_are_checked_but_assignments_are_not(
    computation, valid
):
    outline = SceneOutline(
        scene_id=1,
        title="Arithmetic",
        duration_seconds=1,
        purpose="Show arithmetic",
        math_concept="Equality",
    )
    plan = ScenePlan(
        **outline.model_dump(),
        visual_design="Simple",
        camera_movement="Fixed",
        visual_flow=["Show arithmetic"],
        key_moments=["Equality"],
        computation=computation,
    )
    result = PlanCompiler().compile([outline], [plan], ContinuityBible())
    errors = [
        issue
        for issue in result.issues
        if issue.category == "math" and issue.field.startswith("computation.equation_")
    ]
    assert bool(errors) is not valid
