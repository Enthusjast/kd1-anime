"""Identity-bound results shared by independent pipeline services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from kd1_anime.domain.dependencies import StaleWorkError, WorkIdentity

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StageResult(Generic[T]):
    identity: WorkIdentity
    value: T

    @property
    def revision(self) -> int:
        return self.identity.revision

    @property
    def input_sha256(self) -> str:
        return self.identity.input_sha256

    def accept(self, current: WorkIdentity) -> T:
        if self.identity != current:
            raise StaleWorkError("Stage result belongs to superseded inputs")
        return self.value


def run_stage(identity: WorkIdentity, operation, /, *args, **kwargs) -> StageResult:
    """The service sees explicit values only; dispatch and commit stay outside."""
    return StageResult(identity, operation(*args, **kwargs))
