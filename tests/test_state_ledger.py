import pytest
from pydantic import ValidationError

from kd1_anime.agents.state_ledger import (
    LedgerElement,
    SceneBoundaryIR,
    SceneBoundaryState,
    StateLedger,
    validate_boundary_handoff,
)


def ledger_element(element_id: str = "formula") -> LedgerElement:
    return LedgerElement(
        element_id=element_id,
        variable_name=element_id,
        source_scene_id=1,
        source_code_sha256="a" * 64,
    )


def test_state_ledger_updates_boundary_and_keeps_digest_stable():
    ledger = StateLedger().update_scene(
        scene_id=1,
        elements=[ledger_element()],
        opening_element_ids=[],
        closing_element_ids=["formula"],
        closing_math_state="公式已展示",
    )

    assert ledger.current_scene_id == 1
    assert ledger.boundaries[1].closing_element_ids == ["formula"]
    assert len(ledger.digest()) == 64
    assert ledger.for_elements({"formula"})[0].variable_name == "formula"


def test_state_ledger_uses_boundary_ir_for_transition_contract():
    ledger = StateLedger().update_scene(
        scene_id=1,
        elements=[ledger_element()],
        opening_element_ids=[],
        closing_element_ids=["formula"],
        removed_element_ids=["old_formula"],
        transition_in="从左侧接入",
        transition_out="保留在中心",
        visual_state_digest="a" * 64,
    )

    boundary = ledger.boundaries[1]
    assert isinstance(boundary, SceneBoundaryIR)
    assert boundary.removed_element_ids == ["old_formula"]
    assert boundary.transition_out == "保留在中心"


def test_state_ledger_rejects_boundary_reference_to_unknown_element():
    with pytest.raises(ValidationError, match="不存在的元素"):
        StateLedger(
            elements=[],
            boundaries={
                1: SceneBoundaryState(
                    scene_id=1,
                    closing_element_ids=["missing"],
                )
            },
            current_scene_id=1,
        )


def test_state_ledger_rejects_current_scene_without_boundary():
    with pytest.raises(ValidationError, match="没有对应的场景边界"):
        StateLedger(current_scene_id=1)


def test_boundary_handoff_requires_previous_closing_element():
    current = SceneBoundaryIR(scene_id=2, opening_element_ids=["formula"])

    errors = validate_boundary_handoff(SceneBoundaryIR(scene_id=1), current)

    assert any("closing 元素" in error for error in errors)


def test_boundary_handoff_allows_inherit_then_remove():
    previous = SceneBoundaryIR(scene_id=1, closing_element_ids=["formula"])
    current = SceneBoundaryIR(
        scene_id=2,
        opening_element_ids=["formula"],
        removed_element_ids=["formula"],
    )

    assert validate_boundary_handoff(previous, current) == ()
