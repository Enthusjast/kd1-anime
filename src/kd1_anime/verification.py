"""三层、可持久化的验证结论。

验证结论彼此独立：源码通过静态检查不代表实际运行成功，视频成功
也不代表视觉评估通过。模型故意保留 ``unknown``，避免基础设施或视觉
端点不可用时伪造失败分数。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VerificationStatus = Literal["passed", "failed", "not_run", "unknown"]
ExecutionScope = Literal["import", "frame", "short_video", "formal_video"]
VisualVerificationStatus = Literal[
    "passed",
    "warning",
    "failed",
    "unknown",
    "not_run",
]


class StaticVerification(BaseModel):
    """AST/API/连续性/生命周期检查的收据。"""

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus = "not_run"
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    technical_spec_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    checked_at: str = ""
    error: str = Field(default="", max_length=10_000)


class ExecutionVerification(BaseModel):
    """实际导入、Smoke 或正式视频执行的收据。"""

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus = "not_run"
    scope: ExecutionScope | None = None
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    artifact_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    duration_seconds: float | None = Field(default=None, ge=0)
    checked_at: str = ""
    error: str = Field(default="", max_length=10_000)


class VisualVerification(BaseModel):
    """独立视觉评估的收据；unknown 不等同于低分。"""

    model_config = ConfigDict(extra="forbid")

    status: VisualVerificationStatus = "not_run"
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    artifact_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    report_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    score: float | None = Field(default=None, ge=1.0, le=5.0)
    checked_at: str = ""
    feedback: str = Field(default="", max_length=20_000)


__all__ = [
    "ExecutionScope",
    "ExecutionVerification",
    "StaticVerification",
    "VerificationStatus",
    "VisualVerification",
    "VisualVerificationStatus",
]
