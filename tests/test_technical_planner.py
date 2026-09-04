"""TechnicalSpec 合同和生命周期编译器测试。"""

import pytest

from kd1_anime.agents.planner import ScenePlan, VisualElementState
from kd1_anime.agents.technical_planner import (
    TechnicalAnimation,
    TechnicalLatex,
    TechnicalObject,
    TechnicalSpec,
    compile_technical_spec,
)


def make_plan(*, inherited=None, removed=None, new=None):
    return ScenePlan(
        scene_id=1,
        title="测试场景",
        duration_seconds=10,
        purpose="验证技术合同",
        math_concept="x",
        visual_design="固定布局",
        camera_movement="固定机位",
        visual_flow=["显示公式"],
        key_moments=["结论"],
        computation="x=1",
        inherited_elements=inherited or [],
        elements_to_remove=removed or [],
        new_elements=new or [VisualElementState(element_id="formula", variable_name="formula")],
    )


def test_compile_technical_spec_accepts_create_and_keep_timeline():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="MathTex",
                lifecycle=["define", "fade_in", "keep"],
                exported=True,
            )
        ],
        animations=[
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=0,
                end_seconds=2,
                operation="fade_in",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
            ),
            TechnicalAnimation(
                event_id="hold_formula",
                start_seconds=2,
                end_seconds=10,
                operation="keep",
                source_element_ids=["formula"],
            ),
        ],
        latex=TechnicalLatex(required=True, preamble_packages=["ctex"]),
        export_element_ids=["formula"],
    )

    result = compile_technical_spec(plan, spec)

    assert result.is_valid is True
    assert result.errors == ()


def test_compile_technical_spec_catches_transform_after_source_was_removed():
    old = VisualElementState(element_id="old", variable_name="old")
    new = VisualElementState(element_id="new", variable_name="new")
    plan = make_plan(inherited=[old], removed=[old], new=[new])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(element_id="old", variable_name="old", initially_active=True),
            TechnicalObject(element_id="new", variable_name="new", exported=True),
        ],
        animations=[
            TechnicalAnimation(
                event_id="remove_old",
                start_seconds=0,
                end_seconds=1,
                operation="fade_out",
                source_element_ids=["old"],
            ),
            TechnicalAnimation(
                event_id="transform_old",
                start_seconds=1,
                end_seconds=2,
                operation="transform",
                source_element_ids=["old"],
                target_element_ids=["new"],
            ),
        ],
        export_element_ids=["new"],
    )

    result = compile_technical_spec(plan, spec)

    assert result.is_valid is False
    assert any("尚未 active" in error for error in result.errors)


def test_compile_technical_spec_rejects_unknown_animation_reference():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula", exported=True)],
        animations=[
            TechnicalAnimation(
                event_id="bad",
                start_seconds=0,
                end_seconds=1,
                operation="fade_in",
                target_element_ids=["missing"],
                create_element_ids=["missing"],
            )
        ],
        export_element_ids=["formula"],
    )

    result = compile_technical_spec(plan, spec)

    assert result.is_valid is False
    assert any("未定义对象" in error for error in result.errors)


def test_compile_technical_spec_rejects_camera_frame_for_opengl():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[TechnicalObject(element_id="formula", variable_name="formula", exported=True)],
        export_element_ids=["formula"],
    )
    spec.layout.strategy = "使用 camera.frame 推近"

    result = compile_technical_spec(plan, spec, renderer="opengl")

    assert result.is_valid is False
    assert any("camera.frame" in error for error in result.errors)


def test_compile_technical_spec_requires_xelatex_contract_for_mathtex():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="MathTex",
                exported=True,
            )
        ],
        export_element_ids=["formula"],
    )

    result = compile_technical_spec(plan, spec)

    assert result.is_valid is False
    assert any("latex.required" in error for error in result.errors)


def test_technical_spec_is_closed():
    with pytest.raises(ValueError):
        TechnicalSpec(scene_id=1, unknown_field=True)
