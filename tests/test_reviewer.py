import pytest
from pydantic import ValidationError

from kd1_anime.agents.base import TruncatedResponseError
from kd1_anime.agents.reviewer import (
    REVIEWER_SYSTEM_PROMPT,
    FixSuggestion,
    ReviewFinding,
    ReviewResult,
    normalize_review_evidence,
    validate_review_evidence,
)


def test_reviewer_prompt_contains_real_checklist():
    assert "(略" not in REVIEWER_SYSTEM_PROMPT
    assert "安全边界" in REVIEWER_SYSTEM_PROMPT
    assert "LaTeX" in REVIEWER_SYSTEM_PROMPT
    assert "xelatex" in REVIEWER_SYSTEM_PROMPT
    assert "config.tex_template" in REVIEWER_SYSTEM_PROMPT
    assert "E 类问题一律不阻塞" in REVIEWER_SYSTEM_PROMPT
    assert "fixes 必须可匹配" in REVIEWER_SYSTEM_PROMPT
    assert "相对定位" in REVIEWER_SYSTEM_PROMPT
    assert "KD1_CONTINUITY_EXPORT_BEGIN" in REVIEWER_SYSTEM_PROMPT
    assert "elements_to_remove" in REVIEWER_SYSTEM_PROMPT
    assert "GlobalVisualState" in REVIEWER_SYSTEM_PROMPT
    assert "面积守恒" in REVIEWER_SYSTEM_PROMPT
    assert "保守教学方案" in REVIEWER_SYSTEM_PROMPT
    assert "evidence" in REVIEWER_SYSTEM_PROMPT
    assert "ThreeDScene" in REVIEWER_SYSTEM_PROMPT
    assert "结构化" in REVIEWER_SYSTEM_PROMPT
    assert "证据优先于行号" in REVIEWER_SYSTEM_PROMPT
    assert "只报告确定的问题" in REVIEWER_SYSTEM_PROMPT


def test_severity_is_closed_enum():
    with pytest.raises(ValidationError):
        ReviewResult(is_valid=False, severity="unexpected", feedback="x")


def test_minor_without_fixes_upgrades_to_major():
    result = ReviewResult(is_valid=False, severity="minor", feedback="some issue")
    assert result.severity == "major"


def test_major_requires_feedback():
    with pytest.raises(ValidationError):
        ReviewResult(is_valid=False, severity="major")


def test_valid_result_is_normalized():
    result = ReviewResult(
        is_valid=True,
        severity="major",
        feedback="ignored",
        fixes=[FixSuggestion(find="a", replace="b")],
    )
    assert result.severity == "info"
    assert result.feedback == ""
    assert result.fixes == []


def test_valid_none_severity_is_normalized_without_retry():
    result = ReviewResult.model_validate({"is_valid": True, "severity": "none"})

    assert result.severity == "info"


def test_review_finding_requires_code_evidence_before_it_can_block():
    result = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=2,
                evidence="self.play(Create(circle))",
                why="对象未定义",
                repair="先定义 circle",
            )
        ],
    )

    assert (
        validate_review_evidence(
            result,
            "from manim import *\nself.play(Create(circle))\n",
        )
        == []
    )


def test_review_finding_without_evidence_is_rejected():
    result = ReviewResult(
        is_valid=False,
        severity="major",
        feedback="存在问题",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=2,
                why="无法确定",
                repair="请检查",
            )
        ],
    )

    assert any("evidence" in error for error in validate_review_evidence(result, "x = 1\n"))


def test_review_finding_line_range_must_contain_evidence():
    result = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=1,
                line_end=1,
                evidence="self.play(Create(circle))",
                why="circle 未定义",
                repair="先定义 circle",
            )
        ],
    )

    errors = validate_review_evidence(
        result,
        "from manim import *\nself.play(Create(circle))\n",
    )

    assert any("行号与 evidence 不匹配" in error for error in errors)


def test_normalize_review_evidence_repairs_unique_line_offset():
    result = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=1,
                line_end=1,
                evidence="self.play(Create(circle))",
                why="circle 未定义",
                repair="先定义 circle",
            )
        ],
    )
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.play(Create(circle))\n"

    normalized, corrections = normalize_review_evidence(result, code)

    assert corrections
    assert normalized.findings[0].line_start == 4
    assert normalized.findings[0].line_end == 4
    assert validate_review_evidence(normalized, code) == []


def test_normalize_review_evidence_does_not_guess_duplicate_or_missing_text():
    duplicate = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=1,
                line_end=1,
                evidence="self.wait()",
                why="重复",
                repair="删除一处",
            )
        ],
    )
    missing = duplicate.model_copy(
        update={"findings": [duplicate.findings[0].model_copy(update={"evidence": "missing"})]}
    )

    normalized_duplicate, duplicate_corrections = normalize_review_evidence(
        duplicate,
        "self.wait()\nself.wait()\n",
    )
    normalized_missing, missing_corrections = normalize_review_evidence(missing, "self.wait()\n")

    assert duplicate_corrections == []
    assert normalized_duplicate == duplicate
    assert missing_corrections == []
    assert normalized_missing == missing


def test_reviewer_accepts_unique_evidence_with_wrong_model_line_numbers(monkeypatch):
    from kd1_anime.agents.planner import ScenePlan
    from kd1_anime.agents.reviewer import ReviewerAgent

    scene_plan = ScenePlan(
        scene_id=1,
        title="行号校正",
        duration_seconds=10,
        purpose="测试",
        math_concept="x",
        visual_design="固定",
        camera_movement="固定",
        visual_flow=["显示"],
        key_moments=["停顿"],
        computation="x=1",
    )
    result = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=1,
                line_end=1,
                evidence="self.play(FadeIn(circle))",
                why="circle 未定义",
                repair="先定义 circle",
            )
        ],
    )
    calls = []
    reviewer = ReviewerAgent()

    def fake_call(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call)
    reviewed = reviewer.review(
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.play(FadeIn(circle))",
        scene_plan,
    )

    assert len(calls) == 1
    assert reviewed.findings[0].line_start == 4
    assert reviewed.findings[0].line_end == 4


def test_reviewer_retries_when_model_finding_has_no_code_evidence(monkeypatch):
    from kd1_anime.agents.planner import ScenePlan
    from kd1_anime.agents.reviewer import ReviewerAgent

    scene_plan = ScenePlan(
        scene_id=1,
        title="证据协议",
        duration_seconds=10,
        purpose="测试",
        math_concept="x",
        visual_design="固定",
        camera_movement="固定",
        visual_flow=["显示"],
        key_moments=["停顿"],
        computation="x=1",
    )
    first = ReviewResult(is_valid=False, severity="major", feedback="缺少证据")
    second = ReviewResult(
        is_valid=False,
        severity="major",
        findings=[
            ReviewFinding(
                category="runtime",
                severity="major",
                line_start=4,
                evidence="self.play(FadeIn(circle))",
                why="circle 未定义",
                repair="先定义 circle",
            )
        ],
    )
    results = iter([first, second])
    calls = []
    reviewer = ReviewerAgent()

    def fake_call(**kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call)
    result = reviewer.review(
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.play(FadeIn(circle))",
        scene_plan,
    )

    assert len(calls) == 2
    assert result.findings[0].evidence in calls[1]["user_message"]


def test_invalid_none_severity_still_requires_major_feedback():
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({"is_valid": False, "severity": "none"})


def test_invalid_info_severity_cannot_bypass_review():
    result = ReviewResult.model_validate(
        {
            "is_valid": False,
            "severity": "info",
            "feedback": "存在未修复问题",
        }
    )

    assert result.is_valid is False
    assert result.severity == "major"


def test_reviewer_receives_complete_scene_plan(monkeypatch):
    from kd1_anime.agents.planner import ContinuityBible, ScenePlan
    from kd1_anime.agents.reviewer import ReviewerAgent

    scene_plan = ScenePlan(
        scene_id=1,
        title="几何解释",
        duration_seconds=10,
        purpose="展示投影",
        math_concept="点积",
        visual_design="两个向量",
        camera_movement="固定镜头",
        visual_flow=["显示投影"],
        key_moments=["结果停顿"],
        computation="a dot b = |a||b|cos(theta)",
    )
    captured = {}
    reviewer = ReviewerAgent()

    def fake_call(**kwargs):
        captured.update(kwargs)
        return ReviewResult(is_valid=True)

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call)
    reviewer.review(
        "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()",
        scene_plan,
        continuity_bible=ContinuityBible(background="#101010"),
    )

    message = captured["user_message"]
    assert "展示投影" in message
    assert "a dot b" in message
    assert "<scene_plan>" in message
    assert "<continuity_bible>" in message


def test_reviewer_receives_safe_fallback_mode(monkeypatch):
    from kd1_anime.agents.planner import ScenePlan
    from kd1_anime.agents.reviewer import ReviewerAgent

    scene_plan = ScenePlan(
        scene_id=1,
        title="保守方案",
        duration_seconds=10,
        purpose="展示关系",
        math_concept="面积关系",
        visual_design="基础图形",
        camera_movement="固定",
        visual_flow=["显示公式"],
        key_moments=["停顿"],
        computation="a²+b²=c²",
    )
    captured = {}
    reviewer = ReviewerAgent()

    def fake_call(**kwargs):
        captured.update(kwargs)
        return ReviewResult(is_valid=True)

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call)
    reviewer.review("from manim import *", scene_plan, safe_fallback=True)

    assert "safe_fallback_mode" in captured["user_message"]


def test_reviewer_retries_with_compact_context_after_truncation(monkeypatch):
    from kd1_anime.agents.planner import ScenePlan
    from kd1_anime.agents.reviewer import ReviewerAgent

    scene_plan = ScenePlan(
        scene_id=1,
        title="长上下文审查",
        duration_seconds=10,
        purpose="测试",
        math_concept="x",
        visual_design="布局",
        camera_movement="固定",
        visual_flow=["显示"],
        key_moments=["停顿"],
        computation="x=1",
    )
    calls = []
    reviewer = ReviewerAgent()

    def fake_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TruncatedResponseError("truncated")
        return ReviewResult(is_valid=True)

    monkeypatch.setattr(reviewer, "call_llm_json", fake_call)
    reviewer.review(
        "from manim import *\nclass Demo(Scene):\n    def construct(self): pass",
        scene_plan,
        inherited_elements_code="x = Circle()\n" * 2_000,
    )

    assert len(calls) == 2
    assert len(calls[1]["user_message"]) < len(calls[0]["user_message"])
    assert "继承元素已在 manim_code 中定义" in calls[1]["user_message"]
