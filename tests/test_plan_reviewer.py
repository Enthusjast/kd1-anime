from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kd1_anime.agents.plan_reviewer import (
    PLAN_REVIEW_PROMPT,
    PlanReviewerAgent,
    PlanReviewResult,
    deterministic_plan_issues,
)
from kd1_anime.agents.planner import ContinuityBible, ScenePlan, VisualElementState


def make_plan() -> ScenePlan:
    return ScenePlan(
        scene_id=2,
        title="面积拼接",
        duration_seconds=20,
        purpose="验证面积关系",
        math_concept="a²+b²=c²",
        visual_design="展示三个正方形和公式",
        camera_movement="固定镜头",
        visual_flow=["依次高亮面积标签", "显示等式"],
        key_moments=["0-10s — 高亮", "10-20s — 结论"],
        computation="a=3,b=4,c=5，面积 9+16=25",
        transition_in="接管上一场景的三角形",
        transition_out="保留公式交给下一场景",
    )


def test_plan_review_result_is_closed_and_requires_issues_on_failure():
    with pytest.raises(ValidationError):
        PlanReviewResult(is_valid=False, severity="major")
    with pytest.raises(ValidationError):
        PlanReviewResult.model_validate(
            {
                "is_valid": False,
                "severity": "major",
                "issues": [
                    {
                        "category": "not-a-category",
                        "message": "bad",
                        "fix_instruction": "fix",
                    }
                ],
            }
        )


def test_deterministic_plan_review_rejects_unverified_geometry():
    plan = make_plan().model_copy(update={"visual_flow": ["切割碎片并无缝拼接到目标正方形"]})

    issues = deterministic_plan_issues(plan, ContinuityBible())

    assert any(issue.category == "geometry" for issue in issues)


def test_deterministic_plan_review_accepts_explicit_geometry_calculation():
    plan = make_plan().model_copy(
        update={
            "visual_flow": ["按照顶点和目标覆盖关系将碎片拼接"],
            "computation": "顶点坐标已列出；碎片面积 9+16=25，目标覆盖面积为25",
        }
    )

    issues = deterministic_plan_issues(plan, ContinuityBible())

    assert not any(issue.category == "geometry" for issue in issues)


def test_deterministic_plan_review_does_not_reject_explicit_safe_fallback():
    plan = make_plan().model_copy(
        update={
            "purpose": "已切换为保守教学表达",
            "visual_flow": ["不执行未经验证的碎片拼接，改用等式展示"],
        }
    )

    issues = deterministic_plan_issues(plan, ContinuityBible(), safe_fallback=True)

    assert not any(issue.category == "geometry" for issue in issues)


@patch("kd1_anime.agents.base.BaseAgent.call_llm")
def test_plan_reviewer_sends_plan_and_deterministic_findings(mock_call_llm):
    mock_call_llm.return_value = '{"is_valid": true, "severity": "info", "issues": []}'
    plan = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="formula",
                    variable_name="formula",
                    color_key="primary",
                )
            ]
        }
    )

    result = PlanReviewerAgent().review(
        plan,
        all_plans=[plan],
        continuity_bible=ContinuityBible(),
        deterministic_issues=deterministic_plan_issues(plan, ContinuityBible()),
    )

    assert result.is_valid is True
    message = mock_call_llm.call_args.kwargs["user_message"]
    assert "<user_request>" in message
    assert "<current_scene_plan>" in message
    assert "<deterministic_findings>" in message
    assert "a²+b²=c²" in message


@patch("kd1_anime.agents.base.BaseAgent.call_llm")
def test_plan_reviewer_marks_safe_fallback_context(mock_call_llm):
    mock_call_llm.return_value = '{"is_valid": true, "severity": "info", "issues": []}'

    PlanReviewerAgent().review(make_plan(), safe_fallback=True)

    assert "safe_fallback_mode" in mock_call_llm.call_args.kwargs["user_message"]


def test_plan_review_prompt_requires_math_and_geometry_validation():
    assert "数学公式" in PLAN_REVIEW_PROMPT
    assert "顶点、面积、旋转和目标覆盖关系" in PLAN_REVIEW_PROMPT
    assert "只输出一个 JSON 对象" in PLAN_REVIEW_PROMPT
