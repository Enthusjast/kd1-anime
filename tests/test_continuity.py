"""全片连续性圣经、场景边界合同和连续性审查测试。"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kd1_anime.agents.continuity import (
    CONTINUITY_REVIEW_PROMPT,
    ContinuityIssue,
    ContinuityReviewerAgent,
    ContinuityReviewResult,
    deterministic_continuity_issues,
    extract_continuity_elements,
)
from kd1_anime.agents.planner import (
    ContinuityBible,
    GlobalVisualState,
    ScenePlan,
    VisualElementState,
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
