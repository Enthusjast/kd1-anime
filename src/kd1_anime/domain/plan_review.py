"""Closed planning, compilation and continuity review contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from kd1_anime.models import ValidatedModel as BaseModel


class PlanReviewIssue(BaseModel):
    """一个可定位到分镜字段的计划问题。"""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "math",
        "geometry",
        "feasibility",
        "timing",
        "continuity",
        "contract",
        "renderer",
        "style",
    ]
    severity: Literal["minor", "major"] = "major"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_type: Literal["deterministic", "calculation", "contract", "uncertain"] = "uncertain"
    evidence: str = Field(default="", max_length=5_000)
    field: str = Field(default="", max_length=100)
    message: str = Field(min_length=1, max_length=5_000)
    fix_instruction: str = Field(min_length=1, max_length=5_000)


class PlanReviewResult(BaseModel):
    """计划审查的闭合 JSON 输出契约。"""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    severity: Literal["info", "minor", "major"] = "minor"
    summary: str = Field(default="", max_length=2_000)
    issues: list[PlanReviewIssue] = Field(default_factory=list, max_length=50)
    warnings: list[PlanReviewIssue] = Field(default_factory=list, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def normalize_severity(cls, data):
        if not isinstance(data, dict):
            return data
        value = data.get("severity")
        if value is None or str(value).strip().lower() in {"", "none", "null", "n/a"}:
            normalized = dict(data)
            normalized["severity"] = "info" if data.get("is_valid") else "major"
            return normalized
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> PlanReviewResult:
        if self.is_valid:
            self.severity = "info"
            if self.issues:
                self.warnings = [*self.warnings, *self.issues][:50]
            self.issues = []
            return self
        if self.severity == "info":
            self.severity = "major"
        if not self.issues:
            if self.warnings:
                self.is_valid = True
                self.severity = "info"
                return self
            raise ValueError("计划审查失败时必须提供 issues")
        return self


class PlanReviewBatchItem(PlanReviewResult):
    """批量计划审查中的一个场景结果。"""

    scene_id: int = Field(ge=1)


class ContinuityIssue(BaseModel):
    """一个可定位到场景的连续性冲突。"""

    model_config = ConfigDict(extra="forbid")

    scene_ids: list[int] = Field(min_length=1, max_length=10)
    category: str = Field(min_length=1, max_length=100)
    severity: Literal["minor", "major"] = "major"
    message: str = Field(min_length=1, max_length=5_000)
    fix_instruction: str = Field(default="", max_length=5_000)
    # 让重规划只改真正冲突的字段；旧版 Reviewer 没有输出此字段时由
    # Orchestrator 根据 message/fix_instruction 里的字段名兼容推断。
    target_fields: list[str] = Field(default_factory=list, max_length=12)


class ContinuityReviewResult(BaseModel):
    """全片连续性审查的闭合输出契约。"""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    summary: str = ""
    issues: list[ContinuityIssue] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def normalize_contract(self) -> ContinuityReviewResult:
        if self.is_valid:
            self.issues = []
        elif not self.issues:
            raise ValueError("连续性审查失败时必须提供 issues")
        return self


class PlanCompilerIssue(BaseModel):
    """确定性计划错误，字段与计划审查输出保持一致。"""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "math",
        "geometry",
        "feasibility",
        "timing",
        "continuity",
        "contract",
        "renderer",
        "style",
    ]
    severity: Literal["minor", "major"] = "major"
    scene_ids: list[int] = Field(default_factory=list, max_length=20)
    field: str = Field(default="", max_length=200)
    message: str = Field(min_length=1, max_length=5_000)
    fix_instruction: str = Field(min_length=1, max_length=5_000)


class PlanCompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    issues: list[PlanCompilerIssue] = Field(default_factory=list, max_length=200)
