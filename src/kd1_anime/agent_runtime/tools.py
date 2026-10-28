"""Declared domain tools with durable reservations and cooperative cancellation.

Adapters MUST hold the owning run lock across lookup/reservation (or atomically
compare-and-reserve in persist_intent), persist every callback before returning,
and redact configured secrets. latest_intent looks up the full action_signature
within a session. Any reserved/dispatched/unknown intent requires reconciliation,
not another dispatch. P4 supplies manifest-bound adapters; there is no legacy
execution or generic shell/filesystem fallback here.

Timeouts are cooperative: tools must checkpoint around bounded external calls and
pass remaining_seconds to their transport. This executor never abandons a thread
and never claims that a deadline has stopped an external side effect.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal, get_args

from kd1_anime.agent_runtime.contracts import (
    ACTION_ADAPTER,
    AgentAction,
    ExecutionIntent,
    RuntimeRecord,
    ToolInvocation,
    utc_now,
)
from kd1_anime.agent_runtime.policy import PolicyDecision, ToolPolicySpec
from kd1_anime.commits import RunPatch, ScenePatch
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.services.results import StageResult

current_invocation: ContextVar[ToolInvocation | None] = ContextVar("agent_invocation", default=None)


def action_signature(action: AgentAction) -> str:
    return hashlib.sha256(
        json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolResult:
    stage: StageResult[RuntimeRecord]
    patch: RunPatch | ScenePatch
    waiting_for_job: bool = False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    policy: ToolPolicySpec
    input_model: type[RuntimeRecord]
    output_model: type[RuntimeRecord]
    execute: Callable[[AgentAction, ToolContext], ToolResult]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        tool = replace(
            tool, policy=ToolPolicySpec.model_validate(tool.policy.model_dump(mode="python"))
        )
        models = get_args(get_args(AgentAction)[0])
        declared = {model.model_fields["kind"].default: model for model in models}
        if tool.input_model is not declared.get(tool.policy.action):
            raise ValueError("Tool input must be its exact declared domain action")
        if not isinstance(tool.output_model, type) or not issubclass(
            tool.output_model, RuntimeRecord
        ):
            raise ValueError("Tool outputs must be closed RuntimeRecord models")
        if tool.policy.action in self._tools:
            raise ValueError("Domain tool already registered")
        self._tools[tool.policy.action] = tool

    def configure(self, action: str, **updates) -> None:
        """Replace only a registered tool's closed policy, never its implementation."""
        tool = self.get(action)
        policy = ToolPolicySpec.model_validate({**tool.policy.model_dump(), **updates})
        self._tools[action] = replace(tool, policy=policy)

    def get(self, action: str) -> ToolDefinition:
        try:
            return self._tools[action]
        except KeyError:
            raise ValueError("Unknown domain tool") from None

    @property
    def policies(self) -> tuple[ToolPolicySpec, ...]:
        return tuple(self._tools[name].policy for name in sorted(self._tools))


@dataclass(frozen=True, slots=True)
class ExecutionAdapters:
    latest_intent: Callable[[AgentAction, str], ExecutionIntent | None]
    persist_intent: Callable[[ExecutionIntent], None]
    persist_invocation: Callable[[ToolInvocation], None]
    consume_grant: Callable[[str, ExecutionIntent], object]
    commit: Callable[[ExecutionIntent, ToolInvocation, ToolResult], int]
    emit_committed: Callable[[ExecutionIntent, ToolInvocation, int], None]
    compensate_submission: Callable[[ExecutionIntent, ToolInvocation, ToolResult], None] | None = (
        None
    )


class ToolCancelled(RuntimeError):
    """The cooperative boundary was cancelled; no claim about external state."""


class ToolContext:
    def __init__(
        self,
        *,
        deadline: float,
        cancelled: Callable[[], bool],
        record_external_id: Callable[[str], None],
    ):
        self.deadline = deadline
        self._cancelled = cancelled
        self._record_external_id = record_external_id

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def checkpoint(self) -> None:
        if self._cancelled():
            raise ToolCancelled("Tool cancellation requested")
        if self.remaining_seconds <= 0:
            raise TimeoutError("Tool cooperative deadline reached")

    def record_external_id(self, external_id: str) -> None:
        # Do not checkpoint here: an obtained external ID must be durable even
        # if cancellation/deadline happened immediately after submission.
        if not isinstance(external_id, str) or not external_id or len(external_id) > 256:
            raise ValueError("External ID must be a bounded nonempty string")
        self._record_external_id(external_id)


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: Literal["completed", "failed", "unknown", "cancelled", "pending"]
    intent: ExecutionIntent | None = None
    invocation: ToolInvocation | None = None
    result: ToolResult | None = None
    reason: str = ""


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, adapters: ExecutionAdapters):
        self.registry = registry
        self.adapters = adapters

    def execute(
        self,
        action: AgentAction,
        session_id: str,
        decision: PolicyDecision,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        deadline: float | None = None,
    ) -> ExecutionOutcome:
        if decision.effect != "allow":
            raise PermissionError("Executor requires an allowed policy decision")
        action = ACTION_ADAPTER.validate_python(action.model_dump(mode="python"))
        tool = self.registry.get(action.kind)
        if type(action) is not tool.input_model or action.work_identity is None:
            raise ValueError("Tool requires its exact identity-bound action")
        if decision.allowed_tools and action.kind not in decision.allowed_tools:
            raise PermissionError("Policy did not authorize this tool")
        if (
            tool.policy.requires_approval
            or (action.kind == "inspect_candidate" and action.proposal is not None)
        ) and decision.grant_id is None:
            raise PermissionError("Tool requires an exact approval grant")
        end = time.monotonic() + tool.policy.timeout_seconds
        if deadline is not None:
            end = min(end, deadline)
        if cancelled() or time.monotonic() >= end:
            return ExecutionOutcome(status="cancelled", reason="cancelled_or_deadline")
        previous = self.adapters.latest_intent(action, session_id)
        if previous is not None:
            if previous.session_id != session_id or action_signature(
                previous.action
            ) != action_signature(action):
                raise ValueError("Intent adapter returned a different action identity")
            if previous.status == "completed":
                return ExecutionOutcome(
                    status="completed", intent=previous, reason="already_completed"
                )
            return ExecutionOutcome(
                status="pending", intent=previous, reason="reconciliation_required"
            )
        intent = ExecutionIntent(session_id=session_id, action=action)
        self.adapters.persist_intent(intent)
        if decision.grant_id is not None:
            self.adapters.consume_grant(decision.grant_id, intent)
        invocation = ToolInvocation(
            intent_id=intent.intent_id,
            action=action.kind,
            resource_id=action.resource_id,
            input_sha256=action.input_sha256,
            work_identity=action.work_identity,
            permission=tool.policy.permission,
        )
        self.adapters.persist_invocation(invocation)

        def record_external_id(external_id: str) -> None:
            nonlocal invocation
            if invocation.external_job_id is not None and invocation.external_job_id != external_id:
                raise ValueError("An invocation cannot change its external ID")
            invocation = invocation.model_copy(update={"external_job_id": external_id})
            self.adapters.persist_invocation(invocation)

        context = ToolContext(
            deadline=end, cancelled=cancelled, record_external_id=record_external_id
        )
        try:
            context.checkpoint()
        except (ToolCancelled, TimeoutError):
            invocation = invocation.model_copy(
                update={"status": "cancelled", "finished_at": utc_now()}
            )
            self.adapters.persist_invocation(invocation)
            return ExecutionOutcome(
                status="cancelled",
                intent=intent,
                invocation=invocation,
                reason="cancelled_or_deadline",
            )
        intent = intent.model_copy(update={"status": "dispatched"})
        self.adapters.persist_intent(intent)
        result = None
        for attempt in range(1, tool.policy.max_attempts + 1):
            invocation = invocation.model_copy(
                update={"status": "running", "attempt": attempt, "started_at": utc_now()}
            )
            self.adapters.persist_invocation(invocation)
            try:
                context.checkpoint()
                token = current_invocation.set(invocation)
                try:
                    result = tool.execute(action, context)
                finally:
                    current_invocation.reset(token)
                if action.kind != "submit_render" or invocation.external_job_id is None:
                    context.checkpoint()
                if not isinstance(result, ToolResult) or not isinstance(result.stage, StageResult):
                    raise TypeError("Tool must return a typed StageResult and patch")
                if not isinstance(result.stage.value, tool.output_model):
                    raise TypeError("Tool returned the wrong output model")
                # Revalidate so model_copy/model_construct cannot bypass output privacy/schema.
                output = tool.output_model.model_validate(
                    result.stage.value.model_dump(mode="python")
                )
                result = replace(result, stage=StageResult(result.stage.identity, output))
                result.stage.accept(action.work_identity)
                if not isinstance(result.patch, (RunPatch, ScenePatch)):
                    raise TypeError("Tool must return a typed commit patch")
                if isinstance(result.patch, ScenePatch) and (
                    result.patch.scene_id != action.scene_id
                    or result.patch.identity != action.work_identity
                ):
                    raise ValueError("Scene patch belongs to another action identity")
                if (
                    isinstance(result.patch, RunPatch)
                    and action.scene_id is not None
                    and any(
                        patch.scene_id == action.scene_id and patch.identity != action.work_identity
                        for patch in result.patch.scenes
                    )
                ):
                    raise ValueError(
                        "Run patch contains a scene result for a different action revision"
                    )
                if (
                    isinstance(result.patch, RunPatch)
                    and action.scene_id is not None
                    and result.patch.scenes
                    and not any(patch.scene_id == action.scene_id for patch in result.patch.scenes)
                ):
                    raise ValueError("Scene worker returned only unrelated scene publications")
                if action.kind == "submit_render" and invocation.external_job_id is None:
                    raise ValueError(
                        "Submission must durably record its external ID before returning"
                    )
                break
            except Exception as error:
                if (
                    tool.policy.side_effect == "none"
                    and isinstance(error, (ConnectionError, TimeoutError))
                    and attempt < tool.policy.max_attempts
                    and not cancelled()
                    and time.monotonic() < end
                ):
                    continue
                status = "unknown" if tool.policy.side_effect != "none" else "failed"
                diagnostic = Diagnostic(
                    code=f"tool.{status}",
                    stage="review",
                    category="execution",
                    run_id=action.run_id,
                    scene_id=action.scene_id,
                    summary="Tool execution did not produce a committed result",
                )
                invocation = invocation.model_copy(
                    update={
                        "status": status,
                        "error_diagnostic": diagnostic,
                        "finished_at": utc_now(),
                    }
                )
                intent = intent.model_copy(update={"status": "unknown"})
                self.adapters.persist_intent(intent)
                self.adapters.persist_invocation(invocation)
                return ExecutionOutcome(
                    status=status, intent=intent, invocation=invocation, reason=status
                )
        try:
            revision = self.adapters.commit(intent, invocation, result)
        except Exception:
            intent = intent.model_copy(update={"status": "unknown"})
            try:
                if (
                    action.kind == "submit_render"
                    and invocation.external_job_id is not None
                    and self.adapters.compensate_submission is not None
                ):
                    # This remains compensation for the authorized submission,
                    # not a scheduler side effect in a persistence adapter.
                    self.adapters.compensate_submission(intent, invocation, result)
            finally:
                self.adapters.persist_intent(intent)
            raise
        invocation = invocation.model_copy(update={"status": "completed", "finished_at": utc_now()})
        intent = intent.model_copy(update={"status": "completed"})
        self.adapters.persist_invocation(invocation)
        self.adapters.persist_intent(intent)
        try:
            self.adapters.emit_committed(intent, invocation, revision)
        except Exception:
            # Manifest and lifecycle are already durable. EventStore.reconcile
            # repairs the notification; repeating execution would be unsafe.
            return ExecutionOutcome(
                status="completed",
                intent=intent,
                invocation=invocation,
                result=result,
                reason="notification_missing",
            )
        return ExecutionOutcome(
            status="completed", intent=intent, invocation=invocation, result=result
        )
