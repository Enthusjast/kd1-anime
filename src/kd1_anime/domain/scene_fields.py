"""Authoritative runtime-field ownership and explicit v10 storage projections."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

SceneGroup = Literal["planning", "code_revision", "execution", "verification", "lifecycle"]
SCENE_GROUPS: tuple[SceneGroup, ...] = (
    "planning",
    "code_revision",
    "execution",
    "verification",
    "lifecycle",
)


@dataclass(frozen=True, slots=True)
class SceneFieldMetadata:
    domain_group: SceneGroup
    stored_names: tuple[str, ...]

    @property
    def snapshot_group(self) -> str:
        # v10 keeps its original key; domain ownership never uses that alias.
        return "code" if self.domain_group == "code_revision" else self.domain_group


def _fields(group: SceneGroup, *names: str) -> dict[str, SceneFieldMetadata]:
    return {name: SceneFieldMetadata(group, (name,)) for name in names}


SCENE_FIELDS = MappingProxyType(
    {
        **_fields(
            "planning",
            "plan",
            "plan_ready",
            "plan_replan_attempts",
            "plan_review_round",
            "plan_reviewed",
            "plan_review_feedback",
            "plan_review_signature",
            "technical_spec",
            "technical_contract_stale",
            "technical_spec_sha256",
            "technical_input_sha256",
            "technical_status",
            "technical_error",
            "capability_contract",
            "capability_status",
        ),
        "code": SceneFieldMetadata("code_revision", ("code_file", "code_sha256")),
        **_fields(
            "code_revision",
            "class_name",
            "review_round",
            "safe_fallback_used",
            "safe_fallback_reason",
            "identical_plan_review_count",
            "unknown_animation_detected",
            "unknown_animation_details",
            "rewrite_feedback",
            "review_signature",
            "identical_review_count",
            "reviewed",
            "inherited_elements_code",
            "exported_elements_code",
            "exported_elements",
            "candidates",
        ),
        **_fields(
            "execution",
            "submission",
            "submission_attempts",
            "stage_attempts",
            "fix_attempts",
            "infra_retries",
            "slurm_job",
            "artifact",
            "rendered",
            "resource_profile",
            "last_error_fp",
            "identical_error_count",
            "last_repair_code_sha256",
            "last_repair_error_fp",
            "stagnant_repair_count",
        ),
        **_fields(
            "verification",
            "static_verification",
            "execution_verification",
            "visual_verification",
            "local_smoke_status",
            "visual_status",
            "visual_fix_attempts",
            "visual_score",
            "visual_report_file",
            "visual_report_sha256",
            "visual_artifact_sha256",
            "visual_feedback",
            "visual_best_candidate",
        ),
        **_fields(
            "lifecycle",
            "revision",
            "input_sha256",
            "give_up",
            "failed",
            "failure_reason",
            "failure_category",
            "phase",
        ),
    }
)
SCENE_STORAGE_FIELDS = MappingProxyType(
    {
        stored_name: metadata
        for metadata in SCENE_FIELDS.values()
        for stored_name in metadata.stored_names
    }
)
