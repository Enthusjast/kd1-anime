import pytest
from pydantic import ValidationError

from kd1_anime.agents.reviewer import REVIEWER_SYSTEM_PROMPT, FixSuggestion, ReviewResult


def test_reviewer_prompt_contains_real_checklist():
    assert "(略" not in REVIEWER_SYSTEM_PROMPT
    assert "安全边界" in REVIEWER_SYSTEM_PROMPT
    assert "LaTeX" in REVIEWER_SYSTEM_PROMPT
    assert "xelatex" in REVIEWER_SYSTEM_PROMPT
    assert "config.tex_template" in REVIEWER_SYSTEM_PROMPT
    assert "E 类问题一律不阻塞" in REVIEWER_SYSTEM_PROMPT
    assert "fixes 必须可匹配" in REVIEWER_SYSTEM_PROMPT
    assert "相对定位" in REVIEWER_SYSTEM_PROMPT


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


def test_invalid_none_severity_still_requires_major_feedback():
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({"is_valid": False, "severity": "none"})


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
