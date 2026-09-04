from kd1_anime.agents.plan_compiler import (
    PlanCompiler,
    expressions_are_equivalent,
    normalize_scene_timeline_contract,
)
from kd1_anime.agents.planner import (
    ElementManifest,
    ExtractedElement,
    GeometrySpec,
    LessonSpec,
    MathClaim,
    SceneHandoff,
    SceneOutline,
    ScenePlan,
    TeachingEdge,
    TeachingGraph,
    TimelineEvent,
    VisualElementState,
)
from kd1_anime.run_store import RunManifest, StoredSceneState


def make_plan(**updates):
    data = {
        "scene_id": 1,
        "title": "公式",
        "duration_seconds": 10,
        "purpose": "展示公式",
        "math_concept": "平方",
        "visual_design": "固定画面",
        "camera_movement": "固定",
        "visual_flow": ["逐步展示"],
        "key_moments": ["结论"],
        "computation": "1+1=2",
    }
    data.update(updates)
    return ScenePlan(**data)


def make_outline(scene_id=1):
    return SceneOutline(
        scene_id=scene_id,
        title=f"场景 {scene_id}",
        duration_seconds=10,
        purpose="教学",
        math_concept="公式",
    )


def test_equivalent_polynomials_are_checked_without_external_cas():
    assert expressions_are_equivalent("(a+b)^2", "a^2+2ab+b^2") is True
    assert expressions_are_equivalent("(a+b)^2", "a^2+b^2") is False
    assert expressions_are_equivalent("-ab + ab", "0") is True
    assert expressions_are_equivalent("a^2-ab+ab-b^2", "a^2-b^2") is True


def test_numeric_matrix_equivalence_is_checked_safely():
    assert expressions_are_equivalent("[[1, 2], [3, 4]]", "[[1,2],[3,4]]") is True
    assert expressions_are_equivalent("[[1, 2], [3, 4]]", "[[1,2],[4,3]]") is False
    assert expressions_are_equivalent("[[1, 2]]", "[[1, 2], [3, 4]]") is False


def test_compiler_reports_non_equivalent_matrix_in_computation():
    plan = make_plan(computation="矩阵变换结果 [[1,2],[3,4]] = [[1,2],[4,3]]")

    result = PlanCompiler().compile_scene(plan)

    assert any("矩阵等式" in issue.message for issue in result)


def test_compiler_checks_numeric_matrix_product_in_computation():
    plan = make_plan(computation="矩阵乘法 [[1,2]] * [[3],[4]] = [[8]]")

    result = PlanCompiler().compile_scene(plan)

    assert any("矩阵乘法" in issue.message for issue in result)


def test_compiler_checks_simple_equations_in_free_form_computation():
    plan = make_plan(computation="展开得到 (a+b)^2 = a^2+b^2")

    result = PlanCompiler().compile([make_outline()], [plan])

    assert any(issue.category == "math" for issue in result.issues)


def test_compiler_uses_sampling_when_symbolic_parser_cannot_decide():
    plan = make_plan(
        math_claims=[
            MathClaim(
                claim_id="fraction_claim",
                statement="x/(x+1)=x",
                expression_before="x/(x+1)",
                expression_after="x",
                relation="equivalent",
            )
        ]
    )

    result = PlanCompiler().compile_scene(plan)

    assert any("采样点" in issue.message for issue in result)


def test_compiler_does_not_treat_geometric_constraints_as_identities():
    plan = make_plan(computation="直角三角形满足 a^2+b^2=c^2")

    result = PlanCompiler().compile([make_outline()], [plan])

    assert not any(issue.category == "math" for issue in result.issues)


def test_compiler_does_not_treat_component_relations_as_identities():
    plan = make_plan(computation=("特征向量方程给出 v1=-v2，取向量 (1,-1)；验证 A*(1,-1)=2*(1,-1)"))

    result = PlanCompiler().compile_scene(plan)

    assert not any(issue.category == "math" for issue in result)


def test_compiler_keeps_middle_dot_equations_intact():
    plan = make_plan(computation="符号规则：a·(-b) = -ab，b·(-b) = -b²")

    result = PlanCompiler().compile_scene(plan)

    assert not any(issue.category == "math" for issue in result)


def test_compiler_rejects_timeline_gap_and_bad_math_claim():
    plan = make_plan(
        timeline=[
            TimelineEvent(event_id="first", start_seconds=1, end_seconds=3, action="开场"),
            TimelineEvent(event_id="last", start_seconds=5, end_seconds=6, action="结论"),
        ],
        math_claims=[
            MathClaim(
                claim_id="wrong",
                statement="a+b=a-b",
                relation="equivalent",
            )
        ],
    )

    result = PlanCompiler().compile([make_outline()], [plan])

    assert result.is_valid is False
    assert {issue.category for issue in result.issues} >= {"timing", "math"}


def test_compiler_rejects_timeline_past_scene_duration():
    plan = make_plan(
        timeline=[TimelineEvent(event_id="late", start_seconds=0, end_seconds=11, action="结论")]
    )

    result = PlanCompiler().compile_scene(plan)

    assert any("超出场景时长" in issue.message for issue in result)


def test_compiler_checks_polygon_area_and_accepts_valid_timeline():
    plan = make_plan(
        timeline=[
            TimelineEvent(event_id="all", start_seconds=0, end_seconds=10, action="完整展示")
        ],
        geometry_specs=[
            GeometrySpec(
                geometry_id="square",
                shape="square",
                vertices=[[0, 0], [2, 0], [2, 2], [0, 2]],
                declared_area=4,
                target_area=4,
                target_description="目标正方形",
            )
        ],
    )

    result = PlanCompiler().compile([make_outline()], [plan])

    assert result.is_valid is True


def test_compiler_does_not_treat_line_as_polygon():
    plan = make_plan(
        geometry_specs=[
            GeometrySpec(
                geometry_id="axis",
                shape="line",
                vertices=[[0, 0], [2, 0]],
            )
        ]
    )

    result = PlanCompiler().compile_scene(plan)

    assert result == []


def test_compiler_does_not_apply_2d_shoelace_area_to_3d_region():
    plan = make_plan(
        geometry_specs=[
            GeometrySpec.model_validate(
                {
                    "element_id": "error_region",
                    "type": "region",
                    "vertices": [
                        [1, 1, 2],
                        [1.5, 1, 2.5],
                        [1.5, 1.5, 4.5],
                        [1, 1.5, 3.5],
                    ],
                    "area": 0.5,
                }
            )
        ]
    )

    result = PlanCompiler().compile_scene(plan)

    assert not any(issue.category == "geometry" for issue in result)


def test_compiler_accepts_remove_handoff_for_inherited_element():
    element = VisualElementState(element_id="old", variable_name="old")
    plan = make_plan(
        scene_id=2,
        inherited_elements=[element],
        elements_to_remove=[element],
        handoff=[SceneHandoff(element_id="old", variable_name="old", action="remove")],
    )

    result = PlanCompiler().compile_scene(plan)

    assert not any(issue.field == "handoff" for issue in result)


def _claim_plan(scene_id: int, claim_id: str) -> ScenePlan:
    return make_plan(
        scene_id=scene_id,
        claim_ids=[claim_id],
        timeline=[
            TimelineEvent(
                event_id=f"show_{claim_id}",
                start_seconds=0,
                end_seconds=10,
                action="展示数学关系",
                math_claim_ids=[claim_id],
            )
        ],
        math_claims=[
            MathClaim(
                claim_id=claim_id,
                statement="a=a",
                expression_before="a",
                expression_after="a",
                relation="equivalent",
            )
        ],
    )


def test_compiler_rejects_inherited_element_not_offered_by_previous_scene():
    previous = make_plan(
        scene_id=1,
        new_elements=[VisualElementState(element_id="temporary", variable_name="temporary")],
    )
    current = make_plan(
        scene_id=2,
        inherited_elements=[VisualElementState(element_id="missing", variable_name="missing")],
    )

    result = PlanCompiler().compile([make_outline(1), make_outline(2)], [previous, current])

    continuity_issues = [issue for issue in result.issues if issue.category == "continuity"]
    assert continuity_issues
    assert continuity_issues[0].scene_ids == [2]


def test_compiler_rejects_missing_detail_for_an_outline():
    result = PlanCompiler().compile(
        [make_outline(1), make_outline(2)],
        [make_plan(scene_id=1)],
    )

    assert any(issue.field == "scene_id" for issue in result.issues)


def test_compiler_validates_teaching_graph_and_scene_claim_assignments():
    lesson = LessonSpec(
        claims=[
            MathClaim(claim_id="claim_1", statement="基础", relation="definition"),
            MathClaim(
                claim_id="claim_2",
                statement="结论",
                relation="definition",
                prerequisite_claim_ids=["claim_1"],
            ),
        ]
    )
    graph = TeachingGraph(
        claim_order=["claim_2", "claim_1"],
        edges=[TeachingEdge(prerequisite_claim_id="claim_1", dependent_claim_id="claim_2")],
        scene_claims={1: ["claim_1"], 2: ["claim_2"]},
    )
    result = PlanCompiler().compile(
        [make_outline(1), make_outline(2)],
        [_claim_plan(1, "claim_1"), _claim_plan(2, "claim_2")],
        lesson_spec=lesson,
        teaching_graph=graph,
    )

    assert any(issue.field == "teaching_graph.claim_order" for issue in result.issues)


def test_compiler_accepts_consistent_teaching_graph():
    lesson = LessonSpec(
        claims=[
            MathClaim(claim_id="claim_1", statement="基础", relation="definition"),
            MathClaim(
                claim_id="claim_2",
                statement="结论",
                relation="definition",
                prerequisite_claim_ids=["claim_1"],
            ),
        ]
    )
    graph = TeachingGraph(
        claim_order=["claim_1", "claim_2"],
        edges=[TeachingEdge(prerequisite_claim_id="claim_1", dependent_claim_id="claim_2")],
        scene_claims={1: ["claim_1"], 2: ["claim_2"]},
    )

    result = PlanCompiler().compile(
        [make_outline(1), make_outline(2)],
        [_claim_plan(1, "claim_1"), _claim_plan(2, "claim_2")],
        lesson_spec=lesson,
        teaching_graph=graph,
    )

    assert result.is_valid is True


def test_compiler_requires_each_scene_claim_to_have_detail_and_timeline_evidence():
    plan = make_plan(claim_ids=["claim_1"])

    result = PlanCompiler().compile_scene(plan)

    assert any(issue.field == "math_claims" for issue in result)


def test_compiler_requires_required_new_elements_in_handoff():
    plan = make_plan(
        new_elements=[
            VisualElementState(element_id="result", variable_name="result", required=True)
        ]
    )

    result = PlanCompiler().compile_scene(plan)

    assert any(issue.field == "handoff" and "required=true" in issue.message for issue in result)


def test_compiler_rejects_global_fade_out_of_required_boundary_elements():
    plan = make_plan(
        transition_out="场景结束时所有元素整体淡出",
        closing_state=["所有元素整体淡出"],
        new_elements=[
            VisualElementState(element_id="formula", variable_name="formula", required=True)
        ],
        handoff=[SceneHandoff(element_id="formula", variable_name="formula", action="keep")],
    )

    result = PlanCompiler().compile_scene(plan)

    assert any(issue.field == "transition_out|closing_state" for issue in result)


def test_compiler_allows_exit_deferred_to_next_scene_transition():
    plan = make_plan(
        transition_out="场景结束时保留公式到边界；下一场景淡入时本场景元素整体淡出",
        new_elements=[
            VisualElementState(element_id="formula", variable_name="formula", required=True)
        ],
        handoff=[SceneHandoff(element_id="formula", variable_name="formula", action="keep")],
    )

    result = PlanCompiler().compile_scene(plan)

    assert not any(issue.field == "transition_out|closing_state" for issue in result)


def test_normalize_scene_timeline_contract_absorbs_trailing_pause_before_fade():
    plan = make_plan(
        duration_seconds=10,
        timeline=[
            TimelineEvent(event_id="content", start_seconds=0, end_seconds=5, action="展示结论"),
            TimelineEvent(event_id="fade", start_seconds=5, end_seconds=6, action="整体淡出"),
        ],
    )

    normalized, repairs = normalize_scene_timeline_contract(plan)

    assert normalized.timeline[0].end_seconds == 9
    assert normalized.timeline[1].start_seconds == 9
    assert normalized.timeline[1].end_seconds == 10
    assert repairs


def test_element_manifest_keeps_latest_export_and_dependencies():
    plan = make_plan(
        new_elements=[
            {
                "element_id": "formula",
                "variable_name": "formula",
                "kind": "formula",
                "semantic_state": "结论",
            }
        ]
    )
    manifest = ElementManifest().update_scene(
        plan,
        [
            ExtractedElement(
                element_id="formula",
                variable_name="formula",
                code="formula = MathTex(r'a^2')",
            )
        ],
    )

    assert manifest.entries[0].element_id == "formula"
    assert manifest.entries[0].source_scene_id == 1
    assert manifest.entries[0].source_code_sha256
    assert manifest.scene_exports == {1: ["formula"]}

    stored_manifest = RunManifest(
        run_id="20260828-120000-1234abcd",
        user_prompt="test",
        output_path="/tmp/output.mp4",
        scenes={1: StoredSceneState(plan=plan)},
        element_manifest=manifest,
    )
    assert stored_manifest.integrity_errors() == []
