"""Code rollback candidates shared by runtime state and durable manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from kd1_anime.domain.planning import ExtractedElement
from kd1_anime.models import ValidatedModel
from kd1_anime.rendering import SceneArtifact

CandidateVerification = Literal["validated", "smoke", "rendered"]


class CodeCandidate(ValidatedModel):
    """场景可回滚的代码候选；代码本体保存在 run 内私有文件。"""

    model_config = ConfigDict(extra="forbid")

    code_file: str
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_name: str = Field(min_length=1, max_length=200)
    verification: CandidateVerification = "validated"
    inherited_elements_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    exported_elements_code: str = Field(default="", max_length=30_000)
    exported_elements: list[ExtractedElement] = Field(default_factory=list, max_length=100)
    artifact: SceneArtifact | None = None
    visual_score: float | None = Field(default=None, ge=1.0, le=5.0)
