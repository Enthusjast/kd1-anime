"""Stable generation-pack identity, distinct from each request's final-message hash."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from kd1_anime.agent_runtime.contracts import ACTION_ADAPTER
from kd1_anime.domain.dependencies import digest
from kd1_anime.domain.planning import ContinuityBible, LessonSpec, ScenePlan, TeachingGraph
from kd1_anime.domain.review import ReviewResult
from kd1_anime.domain.runtime import file_digest
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.models import ValidatedModel


class GenerationIdentity(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    prompt_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_builder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


def capture_generation_identity() -> GenerationIdentity:
    """Hash installed implementation bytes and closed response schemas, never credentials.

    Missing source is an incompatibility, not an identity guessed from a version
    string. Wheels include Python sources; editable/source installs use the same
    relative paths, without embedding workstation paths into the digest.
    """
    root = Path(__file__).parent

    def files_hash(paths: Sequence[Path]) -> str:
        return digest(
            {path.relative_to(root).as_posix(): file_digest(path) for path in sorted(paths)}
        )

    agents = sorted((root / "agents").glob("*.py"))
    if not agents:
        raise ValueError("Generation prompt pack source is missing")
    schemas = {
        model.__name__: model.model_json_schema()
        for model in (
            ContinuityBible,
            LessonSpec,
            ScenePlan,
            TeachingGraph,
            TechnicalSpec,
            ReviewResult,
        )
    }
    schemas["AgentAction"] = ACTION_ADAPTER.json_schema()
    return GenerationIdentity(
        prompt_pack_sha256=files_hash(agents),
        response_schema_sha256=digest(schemas),
        context_builder_sha256=file_digest(root / "agents" / "prompt_context.py"),
        generator_sha256=files_hash(
            [
                root / "services" / "generation.py",
                root / "services" / "program_compiler.py",
                root / "agents" / "coder.py",
                root / "prompt_identity.py",
            ]
        ),
        validator_sha256=files_hash(
            [
                root / "candidate_acceptor.py",
                root / "agents" / "validator.py",
                root / "agents" / "api_linter.py",
                root / "agents" / "lifecycle.py",
                root / "agents" / "continuity.py",
                root / "agents" / "capability.py",
                root / "api_catalog.py",
                root / "manim_catalog_0201.py",
            ]
        ),
        policy_sha256=files_hash(
            [
                root / "agent_runtime" / "contracts.py",
                root / "agent_runtime" / "policy.py",
                root / "agent_runtime" / "tools.py",
                root / "agent_runtime" / "loop.py",
                root / "domain" / "execution.py",
                root / "commits.py",
            ]
        ),
    )


def verify_generation_identity(recorded: GenerationIdentity | None) -> None:
    if recorded is None:
        raise ValueError(
            "Run has no verifiable generation identity; legacy identity cannot be fabricated"
        )
    actual = capture_generation_identity()
    mismatched = [
        field
        for field in GenerationIdentity.model_fields
        if getattr(recorded, field) != getattr(actual, field)
    ]
    if mismatched:
        raise ValueError("Incompatible generation identity: " + ", ".join(mismatched))


def final_messages_sha256(messages: Sequence[Mapping]) -> str:
    """Digest the exact final request envelope without storing sensitive bodies."""
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
    roles = {"system", "developer", "user", "assistant", "tool"}
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or set(message) - allowed
            or message.get("role") not in roles
        ):
            raise ValueError("Unsupported final message envelope")
    encoded = json.dumps(
        list(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
