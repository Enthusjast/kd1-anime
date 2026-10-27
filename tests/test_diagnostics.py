"""Behavioral contracts for immutable, evidence-preserving diagnostic adapters."""

import pytest
from pydantic import ValidationError

from kd1_anime.agents.api_linter import ApiLintResult
from kd1_anime.agents.lifecycle import LifecycleValidationResult
from kd1_anime.agents.render_error_parser import RenderErrorEvidence
from kd1_anime.agents.validator import CodeValidationResult
from kd1_anime.domain.diagnostics import (
    Diagnostic,
    diagnostic_from_api,
    diagnostic_from_ast,
    diagnostic_from_continuity_error,
    diagnostic_from_lifecycle,
    diagnostic_from_plan_issue,
    diagnostic_from_render_error,
    diagnostic_from_review_finding,
    diagnostic_from_visual_issue,
)
from kd1_anime.domain.plan_review import PlanReviewIssue
from kd1_anime.domain.review import ReviewFinding
from kd1_anime.eval.visual_eval import VisualIssue

BINDING = {
    "run_id": "run-42",
    "scene_id": 2,
    "revision": 3,
    "code_sha256": "a" * 64,
    "artifact_sha256": "b" * 64,
    "file": "scene_2.py",
}


def make_diagnostic(**changes):
    return Diagnostic(
        **{"code": "test.issue", "stage": "review", "category": "math", "summary": "x", **changes}
    )


def test_diagnostic_is_closed_frozen_and_validates_copies():
    diagnostic = make_diagnostic()
    with pytest.raises(ValidationError):
        diagnostic.summary = "changed"
    with pytest.raises(ValidationError):
        make_diagnostic(arbitrary="not allowed")
    with pytest.raises(ValidationError):
        diagnostic.model_copy(update={"line_start": 0})
    with pytest.raises(ValidationError):
        make_diagnostic(suggested_action="execute_shell")


@pytest.mark.parametrize(
    "changes",
    [
        {"line_start": 0},
        {"line_end": 0},
        {"line_start": 7, "line_end": 6},
        {"scene_id": 0},
        {"revision": -1},
        {"code_sha256": "short"},
        {"artifact_sha256": "A" * 64},
        {"confidence": 1.01},
        {"confidence": -0.1},
        {"confidence": True},
        {"confidence": "0.5"},
    ],
)
def test_diagnostic_rejects_invalid_locations_and_confidence(changes):
    with pytest.raises(ValidationError):
        make_diagnostic(**changes)


def test_ast_diagnostics_preserve_source_result_and_bind_each_error():
    result = CodeValidationResult(False, ["第 12 行: 禁止调用 eval", "缺少 Manim 导入"], ["SceneA"])
    diagnostics = diagnostic_from_ast(result, **BINDING)
    assert len(diagnostics) == 2
    assert diagnostics[0].line_start == diagnostics[0].line_end == 12
    assert diagnostics[1].line_start is None
    for diagnostic in diagnostics:
        assert diagnostic.stage == "ast"
        assert diagnostic.severity == "major"
        assert diagnostic.confidence == "high"
        assert diagnostic.repair_target == "coder"
        assert diagnostic.evidence == {
            "is_valid": False,
            "errors": result.errors,
            "scene_classes": ["SceneA"],
        }
        for key, value in BINDING.items():
            assert getattr(diagnostic, key) == value
    result.errors.append("later mutation")
    assert len(diagnostics[0].evidence["errors"]) == 2


def test_api_errors_and_warnings_do_not_share_severity_or_invent_locations():
    result = ApiLintResult(
        False, ("第 6 行使用已废弃 API ShowCreation",), ("未验证 CustomAnimation",)
    )
    error, warning = diagnostic_from_api(result)
    assert (error.stage, error.severity, error.confidence, error.line_start) == (
        "api",
        "major",
        "high",
        6,
    )
    assert warning.severity == "minor"
    assert warning.confidence == "unknown"
    assert warning.line_start is None
    assert error.evidence["warnings"] == list(result.warnings)
    assert warning.evidence["errors"] == list(result.errors)
    assert diagnostic_from_ast(CodeValidationResult(True)) == ()
    assert diagnostic_from_api(ApiLintResult(True)) == ()


def test_lifecycle_preserves_unknown_animations_without_turning_them_into_errors():
    result = LifecycleValidationResult(
        False,
        ("第 20 行 Transform 使用了尚未 active 的对象: shape",),
        ("[unknown-animation] 第 22 行 CustomAnimation 未被识别",),
        ("第 22 行 CustomAnimation",),
    )
    error, warning = diagnostic_from_lifecycle(result)
    assert error.stage == "lifecycle"
    assert error.severity == "major"
    assert error.line_start == 20
    assert warning.severity == "minor"
    assert warning.confidence == "unknown"
    assert warning.evidence["unknown_animations"] == ["第 22 行 CustomAnimation"]
    assert diagnostic_from_lifecycle(LifecycleValidationResult(True)) == ()


def test_continuity_error_preserves_evidence_and_explicit_location():
    diagnostic = diagnostic_from_continuity_error(
        "必需导出元素 vector_a 缺失", line_start=5, line_end=8, **BINDING
    )
    assert diagnostic.category == diagnostic.stage == "continuity"
    assert diagnostic.repair_target == "coder"
    assert diagnostic.line_start == 5 and diagnostic.line_end == 8
    assert diagnostic.evidence == {"message": "必需导出元素 vector_a 缺失"}
    assert diagnostic.code_sha256 == BINDING["code_sha256"]


def test_plan_adapter_keeps_evidence_type_field_confidence_and_fix_routing():
    issue = PlanReviewIssue(
        category="continuity",
        severity="minor",
        confidence="low",
        evidence_type="contract",
        evidence="scene 2 requires vector_a",
        field="required_elements",
        message="连续性合同缺少对象",
        fix_instruction="在计划中声明 vector_a",
    )
    diagnostic = diagnostic_from_plan_issue(issue, **BINDING)
    assert diagnostic.stage == "plan"
    assert diagnostic.repair_target == "continuity"
    assert diagnostic.severity == "minor" and diagnostic.confidence == "low"
    assert diagnostic.evidence == issue.model_dump(mode="json")
    assert diagnostic.hint == issue.fix_instruction
    assert diagnostic.scene_id == 2 and diagnostic.revision == 3


def test_review_adapter_preserves_lines_code_evidence_and_original_reason():
    finding = ReviewFinding(
        category="math",
        severity="major",
        confidence="medium",
        evidence_type="source_code",
        line_start=9,
        line_end=11,
        evidence="        x = np.sqrt(-1)",
        why="实数域中该表达式未定义",
        repair="先约束定义域",
    )
    diagnostic = diagnostic_from_review_finding(finding, **BINDING)
    assert diagnostic.stage == "review" and diagnostic.category == "math"
    assert diagnostic.line_start == 9 and diagnostic.line_end == 11
    assert diagnostic.confidence == "medium"
    assert diagnostic.evidence == finding.model_dump(mode="json")
    assert diagnostic.evidence["evidence"].startswith("        ")
    assert diagnostic.summary == finding.why
    assert diagnostic.hint == finding.repair
    assert diagnostic.repair_target == "coder"


def test_visual_adapter_preserves_fractional_confidence_frames_boundaries_and_unknown_routing():
    issue = VisualIssue(
        category="overlap",
        severity="minor",
        confidence=0.37,
        repair_target="unknown",
        frame_ids=["F01", "F03"],
        boundary_ids=["scene-1-end", "scene-2-start"],
        evidence="两个标签在边界重叠",
        recommendation="检查边界标签位置",
    )
    diagnostic = diagnostic_from_visual_issue(issue, **BINDING)
    assert diagnostic.stage == "visual" and diagnostic.category == "overlap"
    assert diagnostic.confidence == 0.37
    assert diagnostic.frame_ids == ("F01", "F03")
    assert diagnostic.repair_target == "unknown"
    assert diagnostic.evidence == issue.model_dump(mode="json")
    assert "status" not in diagnostic.evidence and "score" not in diagnostic.evidence
    assert diagnostic.artifact_sha256 == BINDING["artifact_sha256"]


def test_render_adapter_retains_fingerprint_traceback_and_source_exactly():
    evidence = RenderErrorEvidence(
        error_type="ValueError",
        message="shape mismatch",
        category="lifecycle",
        file="scene_2.py",
        line=17,
        code_line="        self.play(Flash(empty))",
        source_context="16:         empty = VGroup()\n17:         self.play(Flash(empty))",
        traceback="Traceback (most recent call last):\n  File 'scene_2.py', line 17\nValueError: shape mismatch",
        fingerprint="0123456789abcdef",
    )
    diagnostic = diagnostic_from_render_error(evidence, **BINDING)
    assert diagnostic.stage == "render" and diagnostic.category == "lifecycle"
    assert diagnostic.line_start == diagnostic.line_end == 17
    assert diagnostic.evidence == evidence.to_dict()
    assert diagnostic.evidence["code_line"].startswith("        ")
    assert diagnostic.repair_target == "coder"
    assert diagnostic.revision == 3


@pytest.mark.parametrize("category", ["infrastructure", "timeout", "renderer"])
def test_render_environment_failures_are_not_routed_to_generated_code(category):
    diagnostic = diagnostic_from_render_error(
        RenderErrorEvidence("", "render unavailable", category)
    )
    assert diagnostic.repair_target == "infrastructure"
    assert diagnostic.severity == "major"


def test_unknown_render_evidence_remains_unknown():
    diagnostic = diagnostic_from_render_error(
        RenderErrorEvidence("", "unrecognized output", "unknown")
    )
    assert diagnostic.category == "unknown"
    assert diagnostic.confidence == "unknown"
    assert diagnostic.repair_target == "unknown"


def test_security_sanitizes_all_text_and_nested_evidence_without_mutating_source():
    evidence = {
        "source": "        request('https://host/?api_key=visible-secret')",
        "nested": [{"Authorization": "Bearer bearer-secret", "token": "nested-secret"}],
        "traceback": "RuntimeError: password=oops-secret",
    }
    diagnostic = make_diagnostic(
        summary="Authorization: Bearer summary-secret",
        detail="<think>private chain of thought</think>visible error",
        hint="<analysis>hidden deliberation</analysis>visible fix",
        evidence=evidence,
    )
    dumped = diagnostic.model_dump_json()
    for secret in (
        "visible-secret",
        "bearer-secret",
        "nested-secret",
        "oops-secret",
        "summary-secret",
    ):
        assert secret not in dumped
    assert "private chain of thought" not in dumped and "hidden deliberation" not in dumped
    assert "visible error" in dumped and "visible fix" in dumped
    assert diagnostic.evidence["source"].startswith("        ")
    assert evidence["nested"][0]["token"] == "nested-secret"


@pytest.mark.parametrize(
    "key", ["reasoning_content", "chain-of-thought", "hidden_analysis", "thinking"]
)
def test_hidden_reasoning_keys_are_rejected_recursively(key):
    with pytest.raises(ValidationError):
        make_diagnostic(evidence={"nested": [{key: "must never be persisted"}]})


def test_adapters_redact_explicit_secrets_including_exception_evidence():
    evidence = RenderErrorEvidence(
        "RuntimeError",
        "opaque-private-value failed",
        "python",
        traceback="RuntimeError: opaque-private-value",
    )
    diagnostic = diagnostic_from_render_error(evidence, secrets=("opaque-private-value",))
    assert "opaque-private-value" not in diagnostic.model_dump_json()
    assert "<redacted>" in diagnostic.evidence["traceback"]


def test_unsafe_exception_objects_are_not_stringified_into_diagnostics():
    class UnsafeException(Exception):
        def __str__(self):
            raise AssertionError("must not stringify an exception")

    with pytest.raises(ValidationError):
        make_diagnostic(evidence={"error": UnsafeException()})


@pytest.mark.parametrize(
    "field", ["arbitrary_location", "severity", "summary", "evidence", "stage"]
)
def test_adapter_rejects_unsupported_location_fields_instead_of_silently_dropping_them(field):
    with pytest.raises((TypeError, ValidationError)):
        diagnostic_from_continuity_error("missing element", **{field: "ignored?"})


def test_standalone_unknown_animation_keeps_uncertainty_and_source_without_failing_candidate():
    source = LifecycleValidationResult(True, unknown_animations=("第 15 行 CustomAnimation",))
    (diagnostic,) = diagnostic_from_lifecycle(source)
    assert diagnostic.severity == "info"
    assert diagnostic.confidence == "unknown"
    assert diagnostic.line_start == diagnostic.line_end == 15
    assert diagnostic.evidence["is_valid"] is True
    assert diagnostic.evidence["unknown_animations"] == ["第 15 行 CustomAnimation"]


def test_unclosed_hidden_reasoning_blocks_cannot_escape_sanitization():
    diagnostic = make_diagnostic(detail="visible evidence\n<analysis>private unfinished thought")
    assert "visible evidence" in diagnostic.detail
    assert "private unfinished thought" not in diagnostic.model_dump_json()
