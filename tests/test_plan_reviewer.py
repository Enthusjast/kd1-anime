from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kd1_anime.agents.plan_reviewer import (
    PLAN_REVIEW_PROMPT,
    PlanReviewerAgent,
    PlanReviewIssue,
    PlanReviewResult,
    classify_plan_review_issues,
    deterministic_plan_issues,
    filter_verified_plan_issues,
)
from kd1_anime.agents.planner import (
    ContinuityBible,
    LessonSpec,
    MathClaim,
    SceneHandoff,
    ScenePlan,
    TeachingGraph,
    VisualElementState,
)


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


def test_plan_reviewer_prompt_includes_readonly_teaching_graph(monkeypatch):
    reviewer = PlanReviewerAgent()
    captured = {}

    def fake_call_llm_json(**kwargs):
        captured["kwargs"] = kwargs
        return PlanReviewResult(is_valid=True, severity="info")

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call_llm_json)
    reviewer.review(
        make_plan(),
        user_prompt="解释公式",
        continuity_bible=ContinuityBible(),
        lesson_spec=LessonSpec(
            claims=[MathClaim(claim_id="claim_1", statement="a=a", relation="definition")]
        ),
        teaching_graph=TeachingGraph(claim_order=["claim_1"], scene_claims={2: ["claim_1"]}),
    )

    assert "teaching_graph" in captured["kwargs"]["user_message"]


def test_plan_review_prompt_does_not_use_claim_removal_to_bypass_evidence():
    assert "不能建议删除 claim_ids" in PLAN_REVIEW_PROMPT


def test_plan_review_does_not_keep_false_math_and_new_element_handoff_errors():
    result_formula = VisualElementState(
        element_id="result_formula",
        variable_name="result_formula",
        required=True,
    )
    plan = make_plan().model_copy(
        update={
            "new_elements": [result_formula],
            "handoff": [
                SceneHandoff(
                    element_id="result_formula",
                    variable_name="result_formula",
                    action="keep",
                )
            ],
            "math_claims": [
                MathClaim(
                    claim_id="cancel",
                    statement="-ab + ab = 0",
                    expression_before="-ab + ab",
                    expression_after="0",
                    relation="equivalent",
                )
            ],
        }
    )
    issues = [
        {
            "category": "math",
            "field": "math_claims[cancel]",
            "message": "前后表达式不等价",
            "fix_instruction": "修正",
        },
        {
            "category": "contract",
            "field": "handoff",
            "message": "new_elements 中的 result_formula 应在 inherited_elements 中声明",
            "fix_instruction": "将其移到 inherited_elements",
        },
    ]

    filtered = filter_verified_plan_issues(
        plan,
        [PlanReviewIssue(**issue) for issue in issues],
    )

    assert filtered == []


def test_plan_review_accepts_next_scene_removal_after_current_create_handoff():
    formula = VisualElementState(
        element_id="formula_surface",
        variable_name="formula_surface",
        required=True,
    )
    plan = make_plan().model_copy(
        update={
            "new_elements": [formula],
            "handoff": [
                SceneHandoff(
                    element_id="formula_surface",
                    variable_name="formula_surface",
                    action="create",
                )
            ],
        }
    )
    issue = PlanReviewIssue(
        category="contract",
        field="handoff",
        message=(
            "场景2的 elements_to_remove 中移除了 formula_surface，"
            "但场景1的 handoff 将其列为 create。"
        ),
        fix_instruction="将当前 handoff 改为 remove。",
    )

    assert filter_verified_plan_issues(plan, [issue]) == []


def test_plan_review_only_major_issues_block():
    plan = make_plan()
    result = PlanReviewResult(
        is_valid=False,
        severity="minor",
        summary="建议调整停顿",
        issues=[
            {
                "category": "timing",
                "severity": "minor",
                "field": "key_moments",
                "message": "停顿略短",
                "fix_instruction": "可选地增加停顿",
            }
        ],
    )

    all_issues, blocking, warnings = classify_plan_review_issues(
        plan,
        deterministic_issues=[],
        result=result,
    )

    assert len(all_issues) == 1
    assert blocking == []
    assert len(warnings) == 1


def test_plan_review_drops_model_issue_that_explicitly_says_no_change_needed():
    plan = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary_step",
                    variable_name="temporary_step",
                    required=False,
                )
            ]
        }
    )
    result = PlanReviewResult(
        is_valid=False,
        severity="major",
        summary="计划符合要求",
        issues=[
            {
                "category": "contract",
                "field": "new_elements",
                "message": (
                    "temporary_step 的 required=false 符合要求，作为中间步骤不应标记为 required=true。"
                ),
                "fix_instruction": "无需修改；确保结束时淡出。",
            }
        ],
    )

    _, blocking, warnings = classify_plan_review_issues(
        plan,
        deterministic_issues=[],
        result=result,
    )

    assert blocking == []
    assert warnings == []


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
    assert "minor" in PLAN_REVIEW_PROMPT
    assert "只输出一个 JSON 对象" in PLAN_REVIEW_PROMPT


@patch("kd1_anime.agents.base.BaseAgent.call_llm")
def test_batch_plan_review_returns_one_result_per_scene(mock_call_llm):
    mock_call_llm.return_value = (
        '{"items": ['
        '{"scene_id": 1, "is_valid": true, "severity": "info", "issues": []},'
        '{"scene_id": 2, "is_valid": true, "severity": "info", "issues": []}'
        "]}"
    )
    first = make_plan().model_copy(update={"scene_id": 1})
    second = make_plan().model_copy(update={"scene_id": 2})

    results = PlanReviewerAgent().review_batch([first, second])

    assert set(results) == {1, 2}
    assert all(result.is_valid for result in results.values())
    assert "批量审查补充规则" in mock_call_llm.call_args.kwargs["system_prompt"]
