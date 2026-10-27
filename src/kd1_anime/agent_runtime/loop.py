"""Bounded orchestration only; domain algorithms remain in injected typed tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from kd1_anime.agent_runtime.contracts import ACTION_ADAPTER, AgentAction, RuntimeRecord
from kd1_anime.agent_runtime.policy import Policy, PolicyBudget, PolicySession, PolicySnapshot
from kd1_anime.agent_runtime.tools import ToolExecutor, action_signature


class LoopResult(RuntimeRecord):
    status: Literal["DONE", "NEED_USER", "FAILED", "BUDGET_EXHAUSTED", "CONTINUE"]
    reason: str
    steps: int = Field(ge=0)
    action: AgentAction | None = None


def _progress(snapshot: PolicySnapshot) -> tuple:
    return (
        snapshot.revision,
        snapshot.state,
        snapshot.work_identity,
        tuple(
            (scene.scene_id, scene.phase, scene.work_identity, scene.external_job_id)
            for scene in snapshot.scenes
        ),
        snapshot.waiting_for_approval,
        snapshot.waiting_for_job,
    )


class AgentLoop:
    """Snapshot callbacks read committed state; no private legacy Context access.

    next_action is deterministic and performs no provider I/O. propose_action is
    optional, called explicitly at most once per step, and must reserve its own
    durable provider request budget before making a request. It is never called
    while waiting, denied all tools, exhausted, cancelled, or terminal.
    """

    def __init__(
        self,
        *,
        policy: Policy,
        executor: ToolExecutor,
        snapshot: Callable[[], PolicySnapshot],
        session: Callable[[], PolicySession],
        budget: Callable[[], PolicyBudget],
        next_action: Callable[[PolicySnapshot], AgentAction | None],
        propose_action: Callable[[PolicySnapshot, tuple[str, ...]], AgentAction] | None = None,
        after_step: Callable[[], None] | None = None,
    ):
        self.policy = policy
        self.executor = executor
        self.snapshot = snapshot
        self.session = session
        self.budget = budget
        self.next_action = next_action
        self.propose_action = propose_action
        self.after_step = after_step

    def step(self, *, cancelled: Callable[[], bool] = lambda: False) -> LoopResult:
        """Authorize and dispatch at most one action without fabricating exhaustion."""
        return self.run(cancelled=cancelled, stop_after_step=True)

    def run(
        self,
        *,
        max_steps: int = 100,
        cancelled: Callable[[], bool] = lambda: False,
        deadline: float | None = None,
        stop_after_step: bool = False,
    ) -> LoopResult:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        signatures: set[str] = set()
        steps = 0
        while True:
            if cancelled():
                return LoopResult(status="NEED_USER", reason="cancelled", steps=steps)
            if deadline is not None and time.monotonic() >= deadline:
                return LoopResult(status="BUDGET_EXHAUSTED", reason="deadline", steps=steps)
            snapshot = self.snapshot()
            if snapshot.state == "DONE":
                return LoopResult(status="DONE", reason="done", steps=steps)
            if snapshot.state == "ERROR":
                return LoopResult(status="FAILED", reason="error_state", steps=steps)
            if snapshot.waiting_for_approval or snapshot.waiting_for_job:
                reason = (
                    "waiting_for_approval" if snapshot.waiting_for_approval else "waiting_for_job"
                )
                return LoopResult(status="NEED_USER", reason=reason, steps=steps)
            budget = self.budget()
            if (
                steps >= max_steps
                or budget.steps_remaining == 0
                or (budget.deadline is not None and snapshot.now >= budget.deadline)
            ):
                return LoopResult(status="BUDGET_EXHAUSTED", reason="budget_exhausted", steps=steps)
            session = self.session()
            try:
                action = self.next_action(snapshot)
                if action is None:
                    available = self.policy.available_tools(
                        snapshot, session, budget, self.executor.registry.policies
                    )
                    if not available or self.propose_action is None:
                        return LoopResult(
                            status="NEED_USER", reason="no_available_action", steps=steps
                        )
                    if budget.requests_remaining == 0 or budget.cost_remaining == 0:
                        return LoopResult(
                            status="BUDGET_EXHAUSTED",
                            reason="request_budget_exhausted",
                            steps=steps,
                        )
                    if cancelled() or (deadline is not None and time.monotonic() >= deadline):
                        return LoopResult(
                            status="NEED_USER", reason="cancelled_or_deadline", steps=steps
                        )
                    action = self.propose_action(snapshot, available)
                action = ACTION_ADAPTER.validate_python(
                    action.model_dump(mode="python") if hasattr(action, "model_dump") else action
                )
                tool = self.executor.registry.get(action.kind)
            except Exception:
                return LoopResult(status="FAILED", reason="invalid_action_proposal", steps=steps)
            signature = action_signature(action)
            if signature in signatures:
                return LoopResult(
                    status="FAILED", reason="repeated_action", steps=steps, action=action
                )
            # Provider proposals can change durable budgets or user commands.
            # Re-read all authorities before authorization, not just before execution.
            current = self.snapshot()
            if current.waiting_for_approval or current.waiting_for_job:
                return LoopResult(
                    status="NEED_USER",
                    reason="waiting_for_approval"
                    if current.waiting_for_approval
                    else "waiting_for_job",
                    steps=steps,
                )
            session = self.session()
            decision = self.policy.evaluate(current, action, tool.policy, session, self.budget())
            if decision.effect != "allow":
                status = (
                    "BUDGET_EXHAUSTED" if decision.reason == "budget_exhausted" else "NEED_USER"
                )
                return LoopResult(status=status, reason=decision.reason, steps=steps, action=action)
            signatures.add(signature)
            try:
                outcome = self.executor.execute(
                    action, session.session_id, decision, cancelled=cancelled, deadline=deadline
                )
            except Exception:
                return LoopResult(
                    status="FAILED",
                    reason="execution_persistence_failed",
                    steps=steps,
                    action=action,
                )
            steps += 1
            if outcome.status == "completed" and outcome.result is not None and self.after_step:
                # The observer is deliberately outside execution/persistence error
                # handling: an interrupted client cannot undo a committed stage.
                self.after_step()
            if outcome.status in {"unknown", "pending", "cancelled"}:
                return LoopResult(
                    status="NEED_USER",
                    reason=outcome.reason or outcome.status,
                    steps=steps,
                    action=action,
                )
            if outcome.status == "failed":
                return LoopResult(
                    status="FAILED", reason=outcome.reason, steps=steps, action=action
                )
            if outcome.result is not None and outcome.result.waiting_for_job:
                return LoopResult(
                    status="NEED_USER", reason="waiting_for_job", steps=steps, action=action
                )
            if _progress(self.snapshot()) == _progress(current):
                return LoopResult(status="FAILED", reason="no_progress", steps=steps, action=action)
            if stop_after_step:
                return LoopResult(
                    status="CONTINUE", reason="step_completed", steps=steps, action=action
                )
