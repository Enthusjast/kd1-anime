"""Frozen diagnostic bindings and routing, retaining sanitized source evidence.

Adapters do not approve candidates, execute code, or apply repairs. Original domain
and LLM contracts remain authoritative; these records only provide a common view.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Unpack

from kd1_anime.models import ValidatedModel
from kd1_anime.security import redact_text, redact_value

if TYPE_CHECKING:
    from kd1_anime.agents.api_linter import ApiLintResult
    from kd1_anime.agents.lifecycle import LifecycleValidationResult
    from kd1_anime.agents.render_error_parser import RenderErrorEvidence
    from kd1_anime.agents.validator import CodeValidationResult
    from kd1_anime.domain.plan_review import PlanReviewIssue
    from kd1_anime.domain.review import ReviewFinding
    from kd1_anime.eval.visual_eval import VisualIssue


Confidence = (
    Literal["high", "medium", "low", "unknown"]
    | Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
)
RepairTarget = Literal["planner", "continuity", "coder", "infrastructure", "unknown"]
SuggestedAction = Literal[
    "repair_code", "revise_plan", "repair_continuity", "inspect_infrastructure", "inspect_evidence"
]


class DiagnosticLocation(TypedDict, total=False):
    """The only caller-supplied binding fields accepted by adapters."""

    run_id: str | None
    scene_id: int | None
    revision: int | None
    code_sha256: str | None
    artifact_sha256: str | None
    file: str | None
    line_start: int | None
    line_end: int | None
    frame_ids: tuple[str, ...]


_HIDDEN_KEYS = frozenset(
    {
        "reasoning_content",
        "chain_of_thought",
        "hidden_analysis",
        "analysis",
        "reasoning",
        "thinking",
    }
)
_LOCATION_FIELDS = frozenset(DiagnosticLocation.__annotations__)
_HIDDEN_BLOCK = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning|reasoning_content)\b[^>]*>"
    r".*?(?:</(?P=tag)\s*>|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_LINE = re.compile(r"第\s*(\d+)\s*行|\bline\s+(\d+)\b", flags=re.IGNORECASE)
_HASH = r"^[0-9a-f]{64}$"


def _safe_text(value: str, secrets: tuple[str, ...]) -> str:
    # redact_text deliberately strips surrounding whitespace. Code evidence must
    # retain indentation and line boundaries, so preserve those at this boundary.
    text = _HIDDEN_BLOCK.sub("<redacted>", value)
    if not text.strip():
        return text
    left = len(text) - len(text.lstrip())
    right = len(text.rstrip())
    return text[:left] + redact_text(text[left:right], secrets) + text[right:]


def _safe_value(value: object, secrets: tuple[str, ...] = ()) -> JsonValue:
    """Copy JSON evidence without ever stringifying arbitrary objects/exceptions."""
    if isinstance(value, str):
        return _safe_text(value, secrets)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Diagnostic values must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Diagnostic evidence keys must be strings")
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _HIDDEN_KEYS:
                raise ValueError("Hidden reasoning cannot be stored in diagnostics")
            # Reuse the central sensitive-key policy, without feeding it values
            # that might be unsafe objects or stripping source-code whitespace.
            if redact_value({key: None})[key] == "<redacted>":
                result[_safe_text(key, secrets)] = "<redacted>"
            else:
                result[_safe_text(key, secrets)] = _safe_value(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, secrets) for item in value]
    raise ValueError("Diagnostic evidence must contain only JSON values")


class Diagnostic(ValidatedModel):
    """Closed, source-bound diagnostic record; never a repair authorization.

    Pydantic freezing protects field bindings, not nested containers. Evidence is
    an owned, sanitized JSON snapshot detached from its source, but is not deeply
    immutable; consumers must treat it as read-only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    code: str = Field(min_length=1)
    stage: Literal["ast", "api", "continuity", "lifecycle", "plan", "review", "visual", "render"]
    category: str = Field(min_length=1)
    severity: Literal["info", "minor", "major"] = "major"
    confidence: Confidence = "unknown"
    run_id: str | None = None
    scene_id: int | None = Field(default=None, ge=1)
    revision: int | None = Field(default=None, ge=0)
    code_sha256: str | None = Field(default=None, pattern=_HASH)
    artifact_sha256: str | None = Field(default=None, pattern=_HASH)
    file: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    frame_ids: tuple[str, ...] = ()
    summary: str = Field(min_length=1)
    detail: str = ""
    hint: str = ""
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    repair_target: RepairTarget = "unknown"
    suggested_action: SuggestedAction | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_input(cls, value):
        if isinstance(value, Mapping):
            return _safe_value(value)
        return value

    @model_validator(mode="after")
    def validate_lines(self) -> Diagnostic:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("Diagnostic.line_end cannot precede line_start")
        return self


def _diagnostic(
    *,
    source: Mapping[str, object],
    location: DiagnosticLocation,
    secrets: tuple[str, ...],
    **fields,
) -> Diagnostic:
    if set(location) - _LOCATION_FIELDS:
        raise TypeError("Unsupported diagnostic location field")
    values = {**fields, "evidence": source, **location}
    return Diagnostic.model_validate(_safe_value(values, secrets))


def _message_lines(message: str) -> DiagnosticLocation:
    match = _LINE.search(message)
    if match is None:
        return {}
    line = int(match.group(1) or match.group(2))
    if line < 1:
        return {}
    return {"line_start": line, "line_end": line}


def diagnostic_from_ast(
    result: CodeValidationResult,
    *,
    secrets: tuple[str, ...] = (),
    **location: Unpack[DiagnosticLocation],
) -> tuple[Diagnostic, ...]:
    source = {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "scene_classes": result.scene_classes,
    }
    return tuple(
        _diagnostic(
            code="ast.error",
            stage="ast",
            category="validation",
            severity="major",
            confidence="high",
            summary=message,
            detail=message,
            repair_target="coder",
            source=source,
            location={**_message_lines(message), **location},
            secrets=secrets,
        )
        for message in result.errors
    )


def diagnostic_from_api(
    result: ApiLintResult, *, secrets: tuple[str, ...] = (), **location: Unpack[DiagnosticLocation]
) -> tuple[Diagnostic, ...]:
    source = {"is_valid": result.is_valid, "errors": result.errors, "warnings": result.warnings}
    return tuple(
        _diagnostic(
            code=f"api.{kind}",
            stage="api",
            category="api",
            severity=severity,
            confidence=confidence,
            summary=message,
            detail=message,
            repair_target="coder",
            source=source,
            location={**_message_lines(message), **location},
            secrets=secrets,
        )
        for kind, messages, severity, confidence in (
            ("error", result.errors, "major", "high"),
            ("warning", result.warnings, "minor", "unknown"),
        )
        for message in messages
    )


def diagnostic_from_lifecycle(
    result: LifecycleValidationResult,
    *,
    secrets: tuple[str, ...] = (),
    **location: Unpack[DiagnosticLocation],
) -> tuple[Diagnostic, ...]:
    source = {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "unknown_animations": result.unknown_animations,
    }
    messages = [("error", message, "major", "high") for message in result.errors] + [
        ("warning", message, "minor", "unknown") for message in result.warnings
    ]
    # Normally unknown animations already have a warning. Retain standalone
    # uncertainty as information, rather than inventing a deterministic failure.
    messages.extend(
        ("unknown_animation", message, "info", "unknown")
        for message in result.unknown_animations
        if not any(message in warning for warning in result.warnings)
    )
    return tuple(
        _diagnostic(
            code=f"lifecycle.{kind}",
            stage="lifecycle",
            category="lifecycle",
            severity=severity,
            confidence=confidence,
            summary=message,
            detail=message,
            repair_target="coder",
            source=source,
            location={**_message_lines(message), **location},
            secrets=secrets,
        )
        for kind, message, severity, confidence in messages
    )


def diagnostic_from_continuity_error(
    message: str, *, secrets: tuple[str, ...] = (), **location: Unpack[DiagnosticLocation]
) -> Diagnostic:
    return _diagnostic(
        code="continuity.error",
        stage="continuity",
        category="continuity",
        severity="major",
        confidence="high",
        summary=message,
        detail=message,
        repair_target="coder",
        source={"message": message},
        location={**_message_lines(message), **location},
        secrets=secrets,
    )


def diagnostic_from_plan_issue(
    issue: PlanReviewIssue, *, secrets: tuple[str, ...] = (), **location: Unpack[DiagnosticLocation]
) -> Diagnostic:
    return _diagnostic(
        code=f"plan.{issue.category}",
        stage="plan",
        category=issue.category,
        severity=issue.severity,
        confidence=issue.confidence,
        summary=issue.message,
        detail=issue.evidence,
        hint=issue.fix_instruction,
        repair_target="continuity" if issue.category == "continuity" else "planner",
        source=issue.model_dump(mode="json"),
        location=location,
        secrets=secrets,
    )


def diagnostic_from_review_finding(
    finding: ReviewFinding, *, secrets: tuple[str, ...] = (), **location: Unpack[DiagnosticLocation]
) -> Diagnostic:
    return _diagnostic(
        code=f"review.{finding.category}",
        stage="review",
        category=finding.category,
        severity=finding.severity,
        confidence=finding.confidence,
        summary=finding.why or finding.evidence or finding.repair or "Review finding",
        detail=finding.evidence,
        hint=finding.repair,
        repair_target="continuity" if finding.category == "continuity" else "coder",
        source=finding.model_dump(mode="json"),
        location={"line_start": finding.line_start, "line_end": finding.line_end, **location},
        secrets=secrets,
    )


def diagnostic_from_visual_issue(
    issue: VisualIssue, *, secrets: tuple[str, ...] = (), **location: Unpack[DiagnosticLocation]
) -> Diagnostic:
    # An issue is not an evaluation status/score. Preserve unknown routing and the
    # original fractional confidence instead of inferring a failed visual review.
    return _diagnostic(
        code=f"visual.{issue.category}",
        stage="visual",
        category=issue.category,
        severity=issue.severity,
        confidence=issue.confidence,
        summary=issue.evidence,
        detail=issue.evidence,
        hint=issue.recommendation,
        repair_target=issue.repair_target,
        source=issue.model_dump(mode="json"),
        location={"frame_ids": tuple(issue.frame_ids), **location},
        secrets=secrets,
    )


def diagnostic_from_render_error(
    error: RenderErrorEvidence,
    *,
    secrets: tuple[str, ...] = (),
    **location: Unpack[DiagnosticLocation],
) -> Diagnostic:
    if error.category in {"infrastructure", "timeout", "renderer"}:
        target: RepairTarget = "infrastructure"
    elif error.category == "unknown":
        target = "unknown"
    else:
        target = "coder"
    source = {
        "error_type": error.error_type,
        "message": error.message,
        "category": error.category,
        "file": error.file,
        "line": error.line,
        "code_line": error.code_line,
        "source_context": error.source_context,
        "traceback": error.traceback,
        "fingerprint": error.fingerprint,
    }
    return _diagnostic(
        code=f"render.{error.category}",
        stage="render",
        category=error.category,
        severity="major",
        confidence="unknown" if error.category == "unknown" else "high",
        summary=error.message or error.error_type or "Unclassified render error",
        detail=error.traceback,
        repair_target=target,
        source=source,
        location={
            "file": error.file or None,
            "line_start": error.line,
            "line_end": error.line,
            **location,
        },
        secrets=secrets,
    )
