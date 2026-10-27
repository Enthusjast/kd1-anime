"""Recovery decisions over explicit inputs; state commits belong to the caller."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kd1_anime.candidate_acceptor import CandidateRejected
from kd1_anime.cluster.status import TERMINAL_STATES
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.domain.technical import TechnicalSpec
from kd1_anime.models import snapshot_value, validated_copy


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    scene_id: int
    job_id: str
    rendered: bool
    status: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class LocalRecovery:
    scene_id: int
    job_id: str
    reason: str
    can_restart: bool = False


@dataclass(frozen=True, slots=True)
class RepairInput:
    code: str
    plan: ScenePlan
    error_log: str
    renderer: str
    class_name: str
    technical_spec: TechnicalSpec | None
    manim_version: str = ""


@dataclass(frozen=True, slots=True)
class RepairedCode:
    code: str
    class_name: str
    feedback: str = ""


class RecoveryService:
    @staticmethod
    def detach_unresumable_local_jobs(scenes: Sequence[RecoveryInput]) -> tuple[LocalRecovery, ...]:
        return tuple(
            LocalRecovery(
                scene.scene_id,
                scene.job_id,
                f"恢复时不认领旧本地渲染任务 {scene.job_id}；"
                + (
                    "已确认终态，可使用新尝试重新启动"
                    if scene.status in TERMINAL_STATES
                    else "无法确认原进程已结束，禁止自动重提"
                ),
                can_restart=scene.status in TERMINAL_STATES,
            )
            for scene in scenes
            if not scene.rendered
        )

    @staticmethod
    def repair(
        agent, inputs: RepairInput, options: Mapping[str, Any], *, validator: Callable, acceptor
    ) -> RepairedCode:
        plan = validated_copy(inputs.plan)
        technical = validated_copy(inputs.technical_spec) if inputs.technical_spec else None
        candidate = agent.fix(inputs.code, inputs.error_log, **snapshot_value(dict(options)))
        try:
            accepted = acceptor.accept_candidate(
                candidate,
                plan,
                technical_spec=technical,
                renderer=inputs.renderer,
                expected_class_name=inputs.class_name or None,
                manim_version=inputs.manim_version,
                validator=validator,
            )
        except CandidateRejected as exc:
            return RepairedCode(candidate, "", str(exc))
        return RepairedCode(accepted.code, accepted.class_name)
