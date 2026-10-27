import pytest
from test_pipeline_planning import boundary as boundary
from test_pipeline_planning import output

from kd1_anime.pipeline.planning import PlanningStageError


def test_safe_plan_proposal_is_adopted_without_discarding_its_content(boundary):
    _, calls, _, factory = boundary
    original = output()
    proposal = original.plans[0].model_copy(
        update={"title": "Conservative pause", "visual_flow": ["Keep a fixed view and wait"]}
    )
    revised = factory().revise(
        original, "Use validated conservative plan", "cairo", proposed_plans=(proposal,)
    )
    assert revised.plans[0].title == revised.outlines[0].title == "Conservative pause"
    assert revised.plans[0].visual_flow == proposal.visual_flow
    assert original.plans[0].title == "Pause" and not calls


def test_safe_plan_proposal_cannot_bypass_mathematics_or_scene_identity(boundary):
    _, calls, _, factory = boundary
    original = output()
    invalid = original.plans[0].model_copy(update={"computation": "1+1=3"})
    with pytest.raises(PlanningStageError):
        factory().revise(original, "Use conservative plan", "cairo", proposed_plans=(invalid,))
    foreign = original.plans[0].model_copy(update={"scene_id": 2})
    with pytest.raises(ValueError):
        factory().revise(original, "Use conservative plan", "cairo", proposed_plans=(foreign,))
    assert original.plans[0].computation == "No computation" and not calls
