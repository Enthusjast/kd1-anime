"""TechnicalSpec 合同和生命周期编译器测试。"""

import pytest

from kd1_anime.agents.planner import ScenePlan, VisualElementState
from kd1_anime.agents.technical_planner import (
    TechnicalAnimation,
    TechnicalLatex,
    TechnicalObject,
    TechnicalSpec,
    compile_technical_spec,
    normalize_technical_spec_contract,
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


def test_compile_technical_spec_ignores_negative_camera_guidance_for_opengl():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[TechnicalObject(element_id="formula", variable_name="formula", exported=True)],
        export_element_ids=["formula"],
        implementation_notes=["OpenGL 渲染器，禁止使用 camera.frame 和 MovingCameraScene"],
    )

    result = compile_technical_spec(plan, spec, renderer="opengl")

    assert not any("camera.frame" in error for error in result.errors)


def test_normalize_technical_spec_repairs_common_lifecycle_hallucinations():
    elements = [
        VisualElementState(element_id="title", variable_name="title", required=True),
        VisualElementState(element_id="step", variable_name="step", required=False),
        VisualElementState(element_id="before", variable_name="before", required=False),
        VisualElementState(element_id="after", variable_name="after", required=False),
        VisualElementState(element_id="result", variable_name="result", required=True),
    ]
    plan = make_plan(new=elements)
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[
            TechnicalObject(
                element_id=item.element_id,
                variable_name=item.variable_name,
                exported=item.required,
                constructor="Text",
            )
            for item in elements
        ],
        animations=[
            TechnicalAnimation(
                event_id="show_title",
                start_seconds=0,
                end_seconds=1,
                operation="fade_in",
                target_element_ids=["title"],
                create_element_ids=["title"],
            ),
            TechnicalAnimation(
                event_id="show_before",
                start_seconds=1,
                end_seconds=2,
                operation="fade_in",
                target_element_ids=["before"],
                create_element_ids=["before"],
            ),
            TechnicalAnimation(
                event_id="replace",
                start_seconds=2,
                end_seconds=3,
                operation="transform",
                source_element_ids=["before"],
                target_element_ids=["after"],
                create_element_ids=["after"],
            ),
            TechnicalAnimation(
                event_id="show_result",
                start_seconds=3,
                end_seconds=4,
                operation="fade_in",
                target_element_ids=["result"],
                create_element_ids=["result"],
            ),
            TechnicalAnimation(
                event_id="cleanup",
                start_seconds=4,
                end_seconds=5,
                operation="fade_out",
                remove_element_ids=["step", "before", "after", "title", "result"],
            ),
            TechnicalAnimation(
                event_id="hold",
                start_seconds=5,
                end_seconds=10,
                operation="wait",
            ),
        ],
        export_element_ids=["title", "result"],
        implementation_notes=["OpenGL renderer，禁止使用 camera.frame 和 MovingCameraScene"],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec, renderer="opengl")
    result = compile_technical_spec(plan, normalized, renderer="opengl")

    cleanup = next(item for item in normalized.animations if item.event_id == "cleanup")
    replace = next(item for item in normalized.animations if item.event_id == "replace")
    assert replace.operation == "replacement_transform"
    assert cleanup.remove_element_ids == ["after"]
    assert result.is_valid is True
    assert repairs


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


def test_normalize_technical_spec_copies_scene_removals_and_inherited_activity():
    inherited = VisualElementState(element_id="old", variable_name="old")
    plan = make_plan(inherited=[inherited], removed=[inherited])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="old", variable_name="old")],
        removed_element_ids=[],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)

    assert normalized.removed_element_ids == ["old"]
    assert normalized.objects[0].initially_active is True
    assert repairs


def test_normalize_technical_spec_introduces_missing_required_export():
    formula = VisualElementState(element_id="formula", variable_name="formula", required=True)
    plan = make_plan(new=[formula])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula")],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    introduction = next(
        event
        for event in normalized.animations
        if event.event_id.startswith("ensure_required_exports")
    )
    assert introduction.operation == "fade_in"
    assert introduction.create_element_ids == ["formula"]
    assert normalized.export_element_ids == ["formula"]
    assert result.is_valid is True, result.errors
    assert any("必需导出元素" in repair for repair in repairs)


def test_normalize_technical_spec_filters_stale_animation_claim_ids():
    formula = VisualElementState(element_id="formula", variable_name="formula", required=True)
    plan = make_plan(new=[formula]).model_copy(update={"claim_ids": ["claim_2"]})
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula")],
        animations=[
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=0,
                end_seconds=2,
                operation="fade_in",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
                claim_ids=["claim_1", "claim_2"],
            )
        ],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    assert normalized.animations[0].claim_ids == ["claim_2"]
    assert result.is_valid is True, result.errors
    assert any("动画断言" in repair for repair in repairs)


def test_normalize_technical_spec_introduces_inactive_animate_target():
    highlight = VisualElementState(
        element_id="highlight",
        variable_name="highlight",
        required=True,
    )
    plan = make_plan(new=[highlight])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="highlight", variable_name="highlight")],
        animations=[
            TechnicalAnimation(
                event_id="highlight",
                start_seconds=0,
                end_seconds=1,
                operation="animate",
                target_element_ids=["highlight"],
            )
        ],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    event = next(item for item in normalized.animations if item.event_id == "highlight")
    assert event.operation == "fade_in"
    assert event.create_element_ids == ["highlight"]
    assert result.is_valid is True, result.errors
    assert any("inactive target" in repair for repair in repairs)


def test_normalize_technical_spec_adds_missing_inherited_removal_event():
    inherited = VisualElementState(element_id="old", variable_name="old")
    formula = VisualElementState(element_id="formula", variable_name="formula")
    plan = make_plan(inherited=[inherited], removed=[inherited], new=[formula])
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[
            TechnicalObject(element_id="old", variable_name="old", initially_active=True),
            TechnicalObject(element_id="formula", variable_name="formula"),
        ],
        animations=[
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=0,
                end_seconds=2,
                operation="fade_in",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
            )
        ],
        implementation_notes=["OpenGL 渲染器，禁止使用 camera.frame"],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec, renderer="opengl")
    result = compile_technical_spec(plan, normalized, renderer="opengl")

    removal = next(
        event
        for event in normalized.animations
        if event.event_id.startswith("remove_planned_elements")
    )
    assert removal.operation == "fade_out"
    assert removal.source_element_ids == ["old"]
    assert result.is_valid is True, result.errors
    assert any("补齐 fade_out" in repair for repair in repairs)


def test_normalize_technical_spec_downgrades_create_of_active_target_to_animation():
    point = VisualElementState(element_id="point", variable_name="point")
    formula = VisualElementState(element_id="formula", variable_name="formula")
    plan = make_plan(inherited=[point], new=[formula])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(element_id="point", variable_name="point", initially_active=True),
            TechnicalObject(element_id="formula", variable_name="formula"),
        ],
        animations=[
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=0,
                end_seconds=1,
                operation="fade_in",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
            ),
            TechnicalAnimation(
                event_id="draw_tangent",
                start_seconds=1,
                end_seconds=2,
                operation="create",
                target_element_ids=["point"],
            ),
        ],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    event = next(item for item in normalized.animations if item.event_id == "draw_tangent")
    assert event.operation == "animate"
    assert event.source_element_ids == ["point"]
    assert result.is_valid is True, result.errors
    assert any("重复 active 对象" in repair for repair in repairs)


def test_normalize_technical_spec_turns_empty_exit_into_wait():
    formula = VisualElementState(element_id="formula", variable_name="formula")
    plan = make_plan(new=[formula])
    spec = TechnicalSpec(
        scene_id=1,
        objects=[TechnicalObject(element_id="formula", variable_name="formula")],
        animations=[
            TechnicalAnimation(
                event_id="fade_out_3d",
                start_seconds=0,
                end_seconds=2,
                operation="fade_out",
            ),
            TechnicalAnimation(
                event_id="show_formula",
                start_seconds=2,
                end_seconds=4,
                operation="fade_in",
                target_element_ids=["formula"],
                create_element_ids=["formula"],
            ),
        ],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    assert normalized.animations[0].operation == "wait"
    assert result.is_valid is True, result.errors
    assert any("空退出操作" in repair for repair in repairs)


def test_normalize_technical_spec_drops_stale_objects_after_plan_rewrite():
    plan = make_plan(
        new=[VisualElementState(element_id="transition_title", variable_name="transition_title")]
    )
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(element_id="old_title", variable_name="old_title"),
            TechnicalObject(element_id="transition_title", variable_name="transition_title"),
        ],
        animations=[
            TechnicalAnimation(
                event_id="fade_old_title",
                start_seconds=0,
                end_seconds=1,
                operation="fade_out",
                source_element_ids=["old_title"],
            ),
            TechnicalAnimation(
                event_id="show_transition_title",
                start_seconds=1,
                end_seconds=2,
                operation="fade_in",
                target_element_ids=["transition_title"],
                create_element_ids=["transition_title"],
            ),
        ],
    )

    normalized, repairs = normalize_technical_spec_contract(plan, spec)
    result = compile_technical_spec(plan, normalized)

    assert [item.element_id for item in normalized.objects] == ["transition_title"]
    assert normalized.animations[0].operation == "wait"
    assert not result.errors
    assert any("old_title" in repair for repair in repairs)


def test_compile_technical_spec_rejects_stale_object_declaration():
    plan = make_plan()
    spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(element_id="formula", variable_name="formula"),
            TechnicalObject(element_id="stale", variable_name="stale"),
        ],
    )

    result = compile_technical_spec(plan, spec)

    assert any("未在 ScenePlan 声明" in error for error in result.errors)


def test_technical_spec_is_closed():
    with pytest.raises(ValueError):
        TechnicalSpec(scene_id=1, unknown_field=True)
