from kd1_anime.agents.plan_compiler import PlanCompiler, expressions_are_equivalent
from kd1_anime.agents.planner import (
    ElementManifest,
    ExtractedElement,
    GeometrySpec,
    MathClaim,
    SceneOutline,
    ScenePlan,
    TimelineEvent,
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


def test_compiler_checks_simple_equations_in_free_form_computation():
    plan = make_plan(computation="展开得到 (a+b)^2 = a^2+b^2")

    result = PlanCompiler().compile([make_outline()], [plan])

    assert any(issue.category == "math" for issue in result.issues)


def test_compiler_does_not_treat_geometric_constraints_as_identities():
    plan = make_plan(computation="直角三角形满足 a^2+b^2=c^2")

    result = PlanCompiler().compile([make_outline()], [plan])

    assert not any(issue.category == "math" for issue in result.issues)


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
