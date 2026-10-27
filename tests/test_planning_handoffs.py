"""Technical planning must preserve the exact structured prior boundary."""

from kd1_anime.domain.planning import ScenePlan, VisualElementState
from kd1_anime.domain.technical import TechnicalHandoff, TechnicalObject, TechnicalSpec
from kd1_anime.services.planning import PlanningService


def scene():
    return ScenePlan(
        scene_id=2,
        title="Keep dot",
        duration_seconds=1,
        purpose="Keep dot",
        math_concept="Point",
        visual_design="Dot",
        camera_movement="Fixed",
        visual_flow=["Hold dot"],
        key_moments=["Dot remains"],
        computation="None",
        inherited_elements=[VisualElementState(element_id="dot", variable_name="dot", kind="Dot")],
    )


def previous():
    return TechnicalHandoff(
        source_scene_id=1,
        elements=[
            TechnicalObject(
                element_id="dot", variable_name="dot", constructor="Dot", final_state="at LEFT"
            )
        ],
    )


def test_technical_handoff_accepts_exact_inherited_identity_and_final_state():
    spec = TechnicalSpec(
        scene_id=2,
        objects=[
            TechnicalObject(
                element_id="dot", variable_name="dot", constructor="Dot", initial_state="at LEFT"
            )
        ],
    )
    assert PlanningService.technical_handoff_errors(scene(), spec, previous()) == ()


def test_technical_handoff_rejects_name_type_and_state_drift():
    spec = TechnicalSpec(
        scene_id=2,
        objects=[
            TechnicalObject(
                element_id="dot",
                variable_name="other",
                constructor="Square",
                initial_state="at RIGHT",
            )
        ],
    )
    errors = PlanningService.technical_handoff_errors(scene(), spec, previous())
    assert any("variable_name" in message for message in errors)
    assert any("constructor" in message for message in errors)
    assert any("initial_state" in message for message in errors)


def test_technical_handoff_rejects_missing_upstream_export():
    assert PlanningService.technical_handoff_errors(
        scene(), TechnicalSpec(scene_id=2), TechnicalHandoff(source_scene_id=1)
    )


def test_technical_handoff_rejects_wrong_boundary_source():
    spec = TechnicalSpec(scene_id=2)
    errors = PlanningService.technical_handoff_errors(
        scene(), spec, TechnicalHandoff(source_scene_id=2, elements=previous().elements)
    )
    assert any("source_scene_id" in message for message in errors)
