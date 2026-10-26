"""Closed review contracts shared by agents and execution services."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from kd1_anime.models import ValidatedModel as BaseModel


class FixSuggestion(BaseModel):
    """单条查找替换。"""

    model_config = ConfigDict(extra="forbid")

    find: str = Field(min_length=1)
    replace: str
    reason: str = ""


class ReviewFinding(BaseModel):
    """带代码证据的单条审查发现。"""

    model_config = ConfigDict(extra="forbid")

    category: Literal["runtime", "math", "latex", "lifecycle", "continuity", "layout"]
    severity: Literal["minor", "major"]
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_type: Literal["deterministic", "source_code", "contract", "visual", "uncertain"] = (
        "uncertain"
    )
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    evidence: str = Field(default="", max_length=2_000)
    why: str = Field(default="", max_length=3_000)
    repair: str = Field(default="", max_length=3_000)

    @model_validator(mode="after")
    def validate_lines(self) -> "ReviewFinding":
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("ReviewFinding.line_end 不能小于 line_start")
        return self


class ReviewResult(BaseModel):
    """审查结果。"""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    severity: Literal["info", "minor", "major"] = "minor"
    feedback: str = ""
    fixes: list[FixSuggestion] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_severity(cls, data):
        """兼容模型常见的 ``none``/null，同时保持失败结果的闭合契约。"""
        if not isinstance(data, dict):
            return data
        severity = data.get("severity")
        if severity is None or str(severity).strip().lower() in {"", "none", "null", "n/a"}:
            normalized = dict(data)
            normalized["severity"] = "info" if data.get("is_valid") else "major"
            return normalized
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> "ReviewResult":
        if self.is_valid:
            self.severity = "info"
            self.feedback = ""
            self.fixes = []
            if self.findings:
                self.warnings = [
                    *self.warnings,
                    *[
                        f"[{finding.category}] {finding.why or finding.repair}"
                        for finding in self.findings
                    ],
                ][:20]
            self.findings = []
            return self
        if self.severity == "info":
            # 失败结果不能因为错误的 severity 而绕过代码审查；与
            # PlanReviewResult 一样采取 fail-closed 策略。
            self.severity = "major"
        if not self.findings and self.warnings and not self.fixes:
            self.is_valid = True
            self.severity = "info"
            self.feedback = ""
            return self
        if self.severity == "minor" and not self.fixes:
            # 没有 fixes 的 minor 升级为 major
            self.severity = "major"
        if self.severity == "major" and not self.feedback.strip():
            if not self.findings:
                raise ValueError("major 审查结果必须包含 feedback 或 findings")
            self.feedback = "\n".join(
                f"第 {finding.line_start or '?'} 行: {finding.why or finding.repair}"
                for finding in self.findings
            )
        return self
