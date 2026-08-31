"""全片连续性圣经、场景边界合同和连续性审查测试。"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kd1_anime.agents.continuity import (
    CONTINUITY_REVIEW_PROMPT,
    ContinuityIssue,
    ContinuityReviewerAgent,
    ContinuityReviewResult,
    apply_deterministic_continuity_repairs,
    deterministic_continuity_issues,
    extract_continuity_elements,
    extract_scene_continuity_elements,
    normalize_scene_plan_contract,
    strip_redundant_optional_export_block,
    validate_export_contract,
)
from kd1_anime.agents.planner import (
    ContinuityBible,
    ExtractedElement,
    GlobalVisualState,
    SceneHandoff,
    SceneOutline,
    ScenePlan,
    VisualElementState,
)
from kd1_anime.agents.safe_fallback import (
    build_safe_fallback_plan,
    is_high_confidence_geometry_conflict,
)


def make_plan(scene_id: int, *, opening=None, closing=None) -> ScenePlan:
    return ScenePlan(
        scene_id=scene_id,
        title=f"Scene {scene_id}",
        duration_seconds=10,
        purpose="演示",
        math_concept="变量 x",
        visual_design="统一深色背景",
        camera_movement="固定中景",
        visual_flow=["显示对象"],
        key_moments=["停顿"],
        computation="x=1",
        persistent_elements=["核心公式 x=1"],
        opening_state=opening if opening is not None else ["核心公式 x=1"],
        closing_state=closing if closing is not None else ["核心公式 x=1"],
        transition_in="公式从上一状态变换接入",
        transition_out="保留核心公式并把焦点交给下一场景",
        continuity_references=["背景 #1C1C1C", "x 使用蓝色"],
    )


def test_scene_plan_keeps_legacy_defaults_and_accepts_continuity_contract():
    plan = ScenePlan(
        scene_id=1,
        title="旧计划",
        duration_seconds=10,
        purpose="测试",
        math_concept="x",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="x=1",
    )

    assert plan.opening_state == []
    assert plan.transition_out == ""


def test_continuity_review_result_requires_issues_for_failure():
    with pytest.raises(ValidationError):
        ContinuityReviewResult(is_valid=False)

    result = ContinuityReviewResult(
        is_valid=True,
        issues=[
            ContinuityIssue(
                scene_ids=[1],
                category="style",
                message="ignored because the result passed",
            )
        ],
    )
    assert result.issues == []


def test_deterministic_continuity_check_detects_missing_contract_and_mismatch():
    plans = [
        make_plan(1, closing=["公式 A"], opening=["初始问题"]),
        make_plan(2, opening=["完全不同的对象"], closing=["结论"]),
    ]

    issues = deterministic_continuity_issues(plans, ContinuityBible())

    categories = {issue.category for issue in issues}
    assert "state" in categories
    assert any(issue.scene_ids == [1, 2] for issue in issues)


def test_removing_an_inherited_element_is_not_a_duplicate_declaration():
    plan = make_plan(1)
    plan = plan.model_copy(
        update={
            "new_elements": [VisualElementState(element_id="formula")],
        }
    )
    next_plan = make_plan(2)
    next_plan = next_plan.model_copy(
        update={
            "inherited_elements": [VisualElementState(element_id="formula")],
            "elements_to_remove": [VisualElementState(element_id="formula")],
        }
    )

    issues = deterministic_continuity_issues([plan, next_plan], ContinuityBible())

    assert not any(
        issue.category == "persistent_element" and issue.scene_ids == [2] for issue in issues
    )


def test_new_element_cannot_also_be_removed_or_inherited():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [VisualElementState(element_id="formula")],
            "inherited_elements": [VisualElementState(element_id="formula")],
        }
    )

    issues = deterministic_continuity_issues([plan], ContinuityBible())

    assert any(issue.category == "persistent_element" for issue in issues)


@patch("kd1_anime.agents.base.BaseAgent.call_llm")
def test_continuity_reviewer_sends_bible_and_all_scene_plans(mock_call_llm):
    mock_call_llm.return_value = '{"is_valid": true, "summary": "通过", "issues": []}'
    agent = ContinuityReviewerAgent()
    bible = ContinuityBible()
    plans = [make_plan(1), make_plan(2)]
    result = agent.review(
        bible,
        [],
        plans,
        deterministic_issues=[],
        stream=False,
    )

    assert result.is_valid is True
    user_message = mock_call_llm.call_args.kwargs["user_message"]
    assert "continuity_bible" in user_message
    assert "Scene 1" in user_message
    assert "opening_state" in user_message


def test_continuity_prompt_is_closed_and_actionable():
    assert "transition_in" in CONTINUITY_REVIEW_PROMPT
    assert "transition_out" in CONTINUITY_REVIEW_PROMPT
    assert "不要输出 Markdown 或代码" in CONTINUITY_REVIEW_PROMPT
    assert "顶点、面积和覆盖关系" in CONTINUITY_REVIEW_PROMPT


def test_deterministic_continuity_repairs_authoritative_transition_and_width():
    plan = make_plan(1)
    plan.transition_out = "下一场景从原点开始绘制"
    plan.visual_flow = ["绘制过程中线宽加粗至 6，其他曲线保持不变"]
    bible = ContinuityBible(
        transition_rules=["新对象先右支后左支，排除零点"],
    )

    repaired = apply_deterministic_continuity_repairs(
        plan,
        bible,
        ["transition_out", "visual_flow"],
        next_outline=SceneOutline(
            scene_id=2,
            title="下一场景",
            duration_seconds=10,
            purpose="继续",
            math_concept="x^{-1}",
        ),
    )

    assert "从原点开始绘制" not in repaired.transition_out
    assert "先右支后左支，排除零点" in repaired.transition_out
    assert "线宽保持默认值 4" in repaired.visual_flow[0]


def test_global_visual_state_and_structured_element_contract():
    state = GlobalVisualState(colors={"primary": "#123456"})
    plan = ScenePlan.model_validate(
        {
            **make_plan(1).model_dump(),
            "global_visual_state": state,
            "new_elements": [
                {
                    "element_id": "main_formula",
                    "role": "核心公式",
                    "variable_name": "formula",
                    "color_key": "primary",
                }
            ],
        }
    )
    assert plan.global_visual_state.colors["primary"] == "#123456"
    assert plan.new_elements[0].element_id == "main_formula"


def test_extract_continuity_elements_prefers_marked_export_and_rejects_side_effects():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: main_formula
        formula = MathTex(r"x^2")
        # KD1_CONTINUITY_EXPORT_END
        self.play(Create(formula))
"""
    exported_code, elements = extract_continuity_elements(code)
    assert "formula = MathTex" in exported_code
    assert elements[0].element_id == "main_formula"

    unsafe = code.replace('formula = MathTex(r"x^2")', 'formula = open("secret")')
    with pytest.raises(ValueError, match="禁止"):
        extract_continuity_elements(unsafe)


def test_extract_continuity_elements_accepts_export_at_construct_tail():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x^2")
        # KD1_CONTINUITY_EXPORT_END
"""

    exported_code, elements = extract_continuity_elements(code)

    assert "formula = MathTex" in exported_code
    assert [item.element_id for item in elements] == ["formula"]


def test_extract_continuity_elements_reports_unpaired_markers_cleanly():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        formula = Circle()
"""

    with pytest.raises(ValueError, match="标记不成对"):
        extract_continuity_elements(code)


def test_extract_continuity_elements_without_marker_uses_safe_fallback():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        formula = MathTex(r"x^2", tex_template=tex_template)
        self.play(Write(formula))
"""

    exported_code, elements = extract_continuity_elements(code)

    assert "formula = MathTex" in exported_code
    assert "tex_template =" not in exported_code
    assert [(item.element_id, item.variable_name) for item in elements] == [("formula", "formula")]


def test_extract_continuity_elements_supports_composite_helpers_in_export_group():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        origin = np.array([0, 0, 0])
        # element_id: main_triangle
        side_a = Line(origin, RIGHT)
        side_b = Line(origin, UP)
        triangle = VGroup(side_a, side_b)
        # element_id: label
        label = Text("x")
        # KD1_CONTINUITY_EXPORT_END
        self.add(triangle, label)
"""

    exported_code, elements = extract_continuity_elements(code)

    assert "origin = np.array" in exported_code
    assert [(item.element_id, item.variable_name) for item in elements] == [
        ("main_triangle", "triangle"),
        ("label", "label"),
    ]


def test_extract_continuity_elements_keeps_named_root_with_child_helpers():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: axes_3d
        axes_3d = ThreeDAxes()
        x_label = MathTex(r"x")
        y_label = MathTex(r"y")
        axes_3d.add(x_label, y_label)
        # KD1_CONTINUITY_EXPORT_END
"""

    _, elements = extract_continuity_elements(code)

    assert [(item.element_id, item.variable_name) for item in elements] == [("axes_3d", "axes_3d")]


def test_extract_continuity_elements_accepts_pure_surface_parameter_helper():
    code = """
from manim import *
class Demo(ThreeDScene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: surface
        def paraboloid(u, v):
            x = u
            y = v
            z = x**2 + y**2
            return np.array([x, y, z])
        surface = Surface(paraboloid, u_range=[-1, 1], v_range=[-1, 1])
        # KD1_CONTINUITY_EXPORT_END
"""

    _, elements = extract_continuity_elements(code)

    assert [(item.element_id, item.variable_name) for item in elements] == [("surface", "surface")]
    assert "def paraboloid" in elements[0].code


def test_extract_continuity_elements_accepts_local_styling_and_safe_aliases():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        COLORS = {"text_dark": "#222222"}
        text_dark = COLORS["text_dark"]
        A_BLUE = "#123456"
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: title
        title = Text("title", color=text_dark, t2c={"title": A_BLUE})
        title.to_edge(UP)
        if len(title) >= 1:
            title.set_color(text_dark)
        # KD1_CONTINUITY_EXPORT_END
"""

    exported_code, elements = extract_continuity_elements(code)

    assert 'text_dark = COLORS["text_dark"]' in exported_code
    assert 'A_BLUE = "#123456"' in exported_code
    assert elements[0].element_id == "title"
    assert "title.to_edge(UP)" in elements[0].code


def test_extract_continuity_elements_accepts_tex_subpart_styling():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"a+b")
        formula.get_part_by_tex("a").set_color(BLUE)
        formula.get_part_by_tex("b").set_color(RED)
        # KD1_CONTINUITY_EXPORT_END
"""

    exported_code, elements = extract_continuity_elements(code)

    assert "get_part_by_tex" in exported_code
    assert [(item.element_id, item.variable_name) for item in elements] == [("formula", "formula")]


def test_extract_continuity_elements_drops_context_assignments_from_marker():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        COLORS = {"primary": "#123456"}
        # element_id: formula
        formula = MathTex(r"x^2", tex_template=tex_template, color=COLORS["primary"])
        # KD1_CONTINUITY_EXPORT_END
"""

    exported_code, elements = extract_continuity_elements(code)

    assert "tex_template =" not in exported_code
    assert "COLORS =" not in exported_code
    assert [(item.element_id, item.variable_name) for item in elements] == [("formula", "formula")]


def test_extract_scene_continuity_elements_ignores_unmarked_internal_objects_without_exports():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary_formula",
                    variable_name="temporary_formula",
                    required=False,
                )
            ]
        }
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        temporary_formula = MathTex(r"x^2", tex_template=tex_template)
        self.play(Write(temporary_formula))
"""

    exported_code, elements = extract_scene_continuity_elements(code, plan)

    assert exported_code == ""
    assert elements == []


def test_extract_scene_continuity_elements_drops_declared_optional_marker_objects():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary_formula",
                    variable_name="temporary_formula",
                    required=False,
                )
            ]
        }
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: temporary_formula
        temporary_formula = MathTex(r"x^2")
        # KD1_CONTINUITY_EXPORT_END
"""

    exported_code, elements = extract_scene_continuity_elements(code, plan)

    assert exported_code == ""
    assert elements == []


def test_extract_scene_continuity_elements_keeps_legacy_fallback_without_contract():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x^2")
"""

    exported_code, elements = extract_scene_continuity_elements(code, make_plan(1))

    assert "formula = MathTex" in exported_code
    assert [(item.element_id, item.variable_name) for item in elements] == [("formula", "formula")]


def test_strip_redundant_optional_export_block_removes_duplicate_definitions():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="formula",
                    variable_name="formula",
                    required=False,
                )
            ]
        }
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x^2")
        self.play(Write(formula))
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x^2")
        # KD1_CONTINUITY_EXPORT_END
"""

    stripped = strip_redundant_optional_export_block(code, plan)

    assert "KD1_CONTINUITY_EXPORT_BEGIN" not in stripped
    assert stripped.count('formula = MathTex(r"x^2")') == 1


def test_extract_continuity_elements_rejects_non_whitelisted_local_method():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x")
        formula.unknown_mutation()
        # KD1_CONTINUITY_EXPORT_END
"""

    with pytest.raises(ValueError, match="不允许的方法"):
        extract_continuity_elements(code)


def test_extract_continuity_elements_rejects_external_uppercase_business_variable():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        C_point = np.array([0, 0, 0])
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: dot
        dot = Dot(C_point)
        # KD1_CONTINUITY_EXPORT_END
"""

    with pytest.raises(ValueError, match="未定义变量: C_point"):
        extract_continuity_elements(code)


def test_extract_continuity_elements_rejects_external_variable_dependency():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        formula = MathTex(expression)
        # KD1_CONTINUITY_EXPORT_END
        self.add(formula)
"""

    with pytest.raises(ValueError, match="未定义变量: expression"):
        extract_continuity_elements(code)


def test_export_contract_requires_declared_structured_elements():
    plan = make_plan(1).model_copy(
        update={"new_elements": [VisualElementState(element_id="formula", variable_name="formula")]}
    )
    with pytest.raises(ValueError, match="未出现在连续性导出区"):
        validate_export_contract(plan, [])

    validate_export_contract(
        plan,
        [
            # 只测试合同映射；代码语法由 extract_continuity_elements 单独负责。
            ExtractedElement(
                element_id="formula",
                variable_name="formula",
                code="formula = MathTex(r'x')",
            )
        ],
    )


def test_export_contract_rejects_removed_and_undeclared_elements():
    inherited = VisualElementState(element_id="old_shape", variable_name="old_shape")
    plan = make_plan(2).model_copy(
        update={
            "inherited_elements": [inherited],
            "elements_to_remove": [inherited],
        }
    )
    exported = [
        ExtractedElement(
            element_id="old_shape",
            variable_name="old_shape",
            code="old_shape = Circle()",
        )
    ]

    with pytest.raises(ValueError, match="已移除元素"):
        validate_export_contract(plan, exported)

    declared = plan.model_copy(
        update={
            "elements_to_remove": [],
            "new_elements": [VisualElementState(element_id="new_shape", variable_name="new_shape")],
        }
    )
    with pytest.raises(ValueError, match="未声明元素"):
        validate_export_contract(
            declared,
            [
                ExtractedElement(
                    element_id="other_shape",
                    variable_name="other_shape",
                    code="other_shape = Circle()",
                )
            ],
        )

    optional_plan = make_plan(2).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary",
                    variable_name="temporary",
                    required=False,
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="非交接元素"):
        validate_export_contract(
            optional_plan,
            [
                ExtractedElement(
                    element_id="temporary",
                    variable_name="temporary",
                    code="temporary = Circle()",
                )
            ],
        )


def test_normalize_scene_plan_contract_repairs_mechanical_conflicts():
    inherited = VisualElementState(element_id="shape", variable_name="shape")
    plan = make_plan(2).model_copy(
        update={
            "global_visual_state": GlobalVisualState(background="#FFFFFF"),
            "inherited_elements": [inherited, inherited],
            "elements_to_remove": [VisualElementState(element_id="stale", variable_name="stale")],
            "new_elements": [
                VisualElementState(element_id="shape", variable_name="shape"),
                VisualElementState(
                    element_id="composite",
                    variable_name="composite",
                    color_key="mixed",
                ),
            ],
        }
    )

    normalized, repairs = normalize_scene_plan_contract(plan, ContinuityBible())

    assert repairs
    assert [item.element_id for item in normalized.inherited_elements] == ["shape"]
    assert normalized.elements_to_remove == []
    assert [item.element_id for item in normalized.new_elements] == ["composite"]
    assert normalized.new_elements[0].color_key == "primary"
    assert normalized.global_visual_state == ContinuityBible().global_visual_state


def test_normalize_scene_plan_contract_preserves_semantic_color_aliases():
    bible = ContinuityBible(
        global_visual_state=GlobalVisualState(
            colors={
                "primary_blue": "#1F77B4",
                "secondary_red": "#D62728",
                "highlight_green": "#2CA02C",
                "neutral_black": "#2C3E50",
                "neutral_gray": "#7F8C8D",
                "background": "#F8F9FA",
            }
        )
    )
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="title",
                    variable_name="title",
                    role="场景标题",
                    color_key="neutral",
                ),
                VisualElementState(
                    element_id="step",
                    variable_name="step",
                    role="步骤标签",
                    color_key="gray",
                ),
                VisualElementState(
                    element_id="formula",
                    variable_name="formula",
                    role="公式",
                    color_key="primary",
                ),
                VisualElementState(
                    element_id="result",
                    variable_name="result",
                    role="最终结论",
                    color_key="highlight",
                ),
            ]
        }
    )

    normalized, _ = normalize_scene_plan_contract(plan, bible)
    colors = {item.element_id: item.color_key for item in normalized.new_elements}

    assert colors == {
        "title": "neutral_black",
        "step": "neutral_gray",
        "formula": "primary_blue",
        "result": "highlight_green",
    }


def test_normalize_scene_plan_contract_handles_explicit_full_exit():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="result", variable_name="result", required=True)
            ],
            "handoff": [SceneHandoff(element_id="result", variable_name="result", action="keep")],
            "closing_state": ["所有元素整体淡出，场景结束"],
            "transition_out": "本场景结束时所有元素整体淡出",
        }
    )

    normalized, repairs = normalize_scene_plan_contract(plan, ContinuityBible())

    assert normalized.new_elements[0].required is False
    assert normalized.handoff == []
    assert any("整体退出" in repair for repair in repairs)


def test_normalize_scene_plan_contract_drops_stale_handoff_for_unexported_previous_element():
    previous = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="transition",
                    variable_name="transition",
                    required=False,
                )
            ]
        }
    )
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(
                    element_id="transition",
                    variable_name="transition",
                    required=True,
                )
            ],
            "handoff": [
                SceneHandoff(
                    element_id="transition",
                    variable_name="transition",
                    action="keep",
                )
            ],
        }
    )

    normalized, repairs = normalize_scene_plan_contract(
        current,
        ContinuityBible(),
        previous_plan=previous,
        has_next_scene=False,
    )

    all_ids = {
        item.element_id
        for group in (
            normalized.inherited_elements,
            normalized.elements_to_remove,
            normalized.new_elements,
        )
        for item in group
    }
    assert "transition" not in all_ids
    assert all(item.element_id != "transition" for item in normalized.handoff)
    assert any("过期 handoff" in repair for repair in repairs)


def test_normalize_scene_plan_contract_drops_unknown_remove_handoff():
    current = make_plan(2).model_copy(
        update={
            "handoff": [
                SceneHandoff(
                    element_id="stale_element",
                    variable_name="stale_element",
                    action="remove",
                )
            ]
        }
    )

    normalized, repairs = normalize_scene_plan_contract(current, ContinuityBible())

    assert normalized.handoff == []
    assert normalized.new_elements == []
    assert any("未声明的过期 handoff" in repair for repair in repairs)


def test_normalize_scene_plan_contract_aligns_handoff_action_with_removal():
    inherited = VisualElementState(element_id="old", variable_name="old")
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [inherited],
            "elements_to_remove": [inherited],
            "handoff": [SceneHandoff(element_id="old", variable_name="old", action="keep")],
        }
    )

    normalized, repairs = normalize_scene_plan_contract(current, ContinuityBible())

    assert normalized.handoff[0].action == "remove"
    assert any("动作已规范为 remove" in repair for repair in repairs)


def test_normalize_scene_plan_contract_uses_handoff_for_boundary_elements():
    plan = make_plan(2).model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="final", variable_name="final"),
                VisualElementState(element_id="temporary", variable_name="temporary"),
            ],
            "handoff": [
                SceneHandoff(
                    element_id="final",
                    variable_name="final",
                    action="keep",
                    semantic_state="最终结果",
                ),
                SceneHandoff(
                    element_id="title",
                    variable_name="title",
                    action="keep",
                    semantic_state="场景标题",
                ),
            ],
        }
    )

    normalized, repairs = normalize_scene_plan_contract(plan, ContinuityBible())

    required = {item.element_id: item.required for item in normalized.new_elements}
    assert required == {"final": True, "temporary": False, "title": True}
    assert any("补入 new_elements" in repair for repair in repairs)


def test_deterministic_continuity_check_reports_unknown_color_key():
    plan = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="shape",
                    variable_name="shape",
                    color_key="mixed",
                )
            ]
        }
    )

    issues = deterministic_continuity_issues([plan], ContinuityBible())

    assert any("未定义颜色键" in issue.message for issue in issues)


def test_normalize_scene_plan_contract_drops_unexported_previous_elements():
    previous = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="kept", variable_name="kept"),
            ]
        }
    )
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(element_id="kept", variable_name="kept"),
                VisualElementState(element_id="missing", variable_name="missing"),
            ]
        }
    )

    normalized, repairs = normalize_scene_plan_contract(
        current,
        ContinuityBible(),
        previous_plan=previous,
    )

    assert [item.element_id for item in normalized.inherited_elements] == ["kept"]
    assert any("未声明导出" in repair for repair in repairs)


def test_normalize_scene_plan_contract_drops_all_inherited_when_previous_exports_none():
    previous = make_plan(1)
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(element_id="missing", variable_name="missing")
            ]
        }
    )

    normalized, repairs = normalize_scene_plan_contract(
        current,
        ContinuityBible(),
        previous_plan=previous,
    )

    assert normalized.inherited_elements == []
    assert any("未声明导出" in repair for repair in repairs)


def test_normalize_scene_plan_contract_drops_optional_previous_elements():
    previous = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary",
                    variable_name="temporary",
                    required=False,
                )
            ]
        }
    )
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(element_id="temporary", variable_name="temporary")
            ]
        }
    )

    normalized, repairs = normalize_scene_plan_contract(
        current,
        ContinuityBible(),
        previous_plan=previous,
    )

    assert normalized.inherited_elements == []
    assert any("未声明导出" in repair for repair in repairs)


def test_continuity_state_matching_keeps_single_letter_math_variables():
    plans = [
        make_plan(1, closing=["保留变量 a"]),
        make_plan(2, opening=["接管变量 a"], closing=["结论"]),
    ]

    issues = deterministic_continuity_issues(plans, ContinuityBible())

    assert not any(issue.category == "state" for issue in issues)


def test_normalize_scene_plan_contract_keeps_previous_variable_name():
    previous = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="triangle", variable_name="right_triangle"),
            ]
        }
    )
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(element_id="triangle", variable_name="triangle"),
            ]
        }
    )

    normalized, repairs = normalize_scene_plan_contract(
        current,
        ContinuityBible(),
        previous_plan=previous,
    )

    assert normalized.inherited_elements[0].variable_name == "right_triangle"
    assert any("变量名固定" in repair for repair in repairs)


def test_deterministic_continuity_check_reports_variable_name_drift():
    previous = make_plan(1).model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="triangle", variable_name="right_triangle"),
            ]
        }
    )
    current = make_plan(2).model_copy(
        update={
            "inherited_elements": [
                VisualElementState(element_id="triangle", variable_name="triangle"),
            ]
        }
    )

    issues = deterministic_continuity_issues([previous, current], ContinuityBible())

    assert any(issue.category == "element_handoff" for issue in issues)


def test_safe_fallback_only_targets_high_confidence_geometry_failures():
    plan = make_plan(2).model_copy(
        update={
            "visual_flow": ["切割碎片并无缝拼接到目标正方形"],
            "new_elements": [
                VisualElementState(
                    element_id="reassembled_square",
                    variable_name="reassembled_square",
                    color_key="mixed",
                ),
                VisualElementState(
                    element_id="piece_a",
                    variable_name="piece_a",
                    color_key="primary",
                ),
            ],
        }
    )

    assert is_high_confidence_geometry_conflict(plan, "几何关系未验证，碎片无法覆盖目标区域")
    assert not is_high_confidence_geometry_conflict(make_plan(2), "建议调整布局")

    fallback = build_safe_fallback_plan(
        plan,
        ContinuityBible(),
        reason="碎片目标位置不明确",
    )
    assert "保守" in fallback.purpose
    assert all("拼接" not in step or "不执行" in step for step in fallback.visual_flow)
    assert (
        next(item for item in fallback.new_elements if item.element_id == "piece_a").required
        is False
    )


def test_safe_fallback_detects_geometry_feedback_for_square_plan():
    plan = make_plan(2).model_copy(
        update={"visual_design": "在同一画面构造三个正方形", "computation": "顶点坐标和面积"}
    )

    assert is_high_confidence_geometry_conflict(
        plan,
        "[geometry] 正方形顶点位置错误，几何关系不一致",
    )
