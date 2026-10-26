"""Immutable, secret-free configuration and execution identities for a run."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator

from kd1_anime.models import ValidatedModel


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Execution image changed while computing its identity")
    return digest.hexdigest()


class RunConfig(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    values_json: str

    @field_validator("values_json")
    @classmethod
    def validate_snapshot(cls, value):
        from kd1_anime.config import Settings

        values = json.loads(value)
        expected = {name for name in Settings.model_fields if not name.endswith("API_KEY")}
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError("RunConfig requires all non-sensitive configuration fields")
        # Rebuild every field without reading credential defaults into the snapshot.
        Settings(
            _env_file=None,
            **values,
            **{name: "" for name in Settings.model_fields if name.endswith("API_KEY")},
        )
        return value

    @classmethod
    def capture(cls):
        from kd1_anime.config import settings

        values = settings.model_dump(mode="json")
        secrets = [value for key, value in values.items() if key.endswith("API_KEY") and value]
        values = {key: value for key, value in values.items() if not key.endswith("API_KEY")}
        # An endpoint may contain an explicitly configured credential as well.
        from kd1_anime.security import redact_value

        return cls(
            values_json=json.dumps(
                redact_value(values, tuple(secrets)), sort_keys=True, ensure_ascii=False
            )
        )

    def resolve(self):
        from kd1_anime.config import Settings, settings

        values = json.loads(self.values_json)
        credentials = {
            key: getattr(settings, key) for key in Settings.model_fields if key.endswith("API_KEY")
        }
        return Settings(_env_file=None, **values, **credentials)

    @contextmanager
    def activate(self):
        from kd1_anime.clients import client_scope
        from kd1_anime.config import runtime_settings

        with runtime_settings(self.resolve()), client_scope():
            yield


class ExecutionSpec(ValidatedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    image: str = ""
    image_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    require_container: bool = False
    disable_network: bool = True
    render_json: str = "{}"
    resources_json: str = "{}"
    python_version: str = ""

    @field_validator("render_json")
    @classmethod
    def validate_render(cls, value):
        from kd1_anime.rendering import RenderProfile

        RenderProfile.model_validate_json(value)
        return value

    @field_validator("resources_json")
    @classmethod
    def validate_resources(cls, value):
        from kd1_anime.config import Settings

        data = json.loads(value)
        expected = {
            name
            for name in Settings.model_fields
            if name.startswith(("SLURM_", "LOCAL_RENDER_", "LOCAL_SMOKE_", "SMOKE_"))
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("ExecutionSpec requires a complete resource and isolation policy")
        defaults = {
            name: field.get_default(call_default_factory=True)
            for name, field in Settings.model_fields.items()
        }
        Settings(_env_file=None, **{**defaults, **data})
        return value

    @classmethod
    def capture(cls, profile=None):
        from kd1_anime.config import settings
        from kd1_anime.rendering import RenderProfile

        profile = profile or RenderProfile.current()
        image = (
            str(Path(settings.SLURM_CONTAINER_IMAGE).expanduser().resolve())
            if settings.SLURM_CONTAINER_IMAGE
            else ""
        )
        resources = {
            key: value
            for key, value in settings.model_dump(mode="json").items()
            if key.startswith(("SLURM_", "LOCAL_RENDER_", "LOCAL_SMOKE_", "SMOKE_"))
        }
        return cls(
            image=image,
            image_sha256=file_digest(Path(image)) if image and Path(image).is_file() else "",
            require_container=settings.SLURM_REQUIRE_CONTAINER,
            disable_network=settings.SLURM_CONTAINER_DISABLE_NETWORK,
            render_json=json.dumps(
                profile.model_dump(mode="json", exclude={"execution_sha256"}), sort_keys=True
            ),
            resources_json=json.dumps(resources, sort_keys=True),
            python_version="" if image else sys.version.split()[0],
        )

    def for_profile(self, profile):
        return self.model_copy(
            update={
                "render_json": json.dumps(
                    profile.model_dump(mode="json", exclude={"execution_sha256"}), sort_keys=True
                )
            }
        )

    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def verify_image(self):
        if self.require_container and not self.image:
            raise ValueError("Execution contract requires a container image")
        if self.image and (
            not self.image_sha256 or file_digest(Path(self.image)) != self.image_sha256
        ):
            raise ValueError("Container image differs from the immutable execution contract")

    def matches_environment(self, actual):
        required = ("python", "manim", "ffmpeg", "xelatex", "renderer")
        if any(not actual.get(name) for name in required):
            return False
        if self.python_version and actual.get("python") != self.python_version:
            return False
        if actual.get("execution_sha256") != self.digest():
            return False
        if self.image and (
            not self.image_sha256 or actual.get("container_sha256") != self.image_sha256
        ):
            return False
        expected = json.loads(self.render_json)
        versions = {
            "manim": "manim_version",
            "ffmpeg": "ffmpeg_version",
            "xelatex": "xelatex_version",
            "renderer": "renderer",
        }
        return all(
            not expected.get(field) or actual.get(key) == expected[field]
            for key, field in versions.items()
        )
