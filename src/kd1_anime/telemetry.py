"""Scene spans and request correlation without inferring timing from UI phases."""

from __future__ import annotations

import time
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4

CURRENT_STAGE: ContextVar[str] = ContextVar("pipeline_stage", default="")
CURRENT_SPAN: ContextVar[str] = ContextVar("pipeline_span", default="")


def workflow_span(stage: str):
    """Record run stages and technical stages with their parent correlation."""

    def decorate(method):
        @wraps(method)
        def run(self, ctx, *args, **kwargs):
            state = args[0] if args and hasattr(args[0], "plan") else None
            fields = {
                "span_id": uuid4().hex,
                "parent_span_id": CURRENT_SPAN.get(),
                "stage": stage,
                "scene_id": state.plan.scene_id if state is not None else None,
                "scene_revision": state.revision if state is not None else None,
            }
            stage_token = CURRENT_STAGE.set(stage)
            span_token = CURRENT_SPAN.set(fields["span_id"])
            self._emit("stage_span_started", **fields)
            started = time.monotonic()
            outcome = "completed"
            try:
                return method(self, ctx, *args, **kwargs)
            except BaseException:
                outcome = "error"
                raise
            finally:
                self._emit(
                    "stage_span_finished",
                    **fields,
                    duration_seconds=time.monotonic() - started,
                    outcome=outcome,
                    completed_revision=state.revision if state is not None else None,
                )
                CURRENT_SPAN.reset(span_token)
                CURRENT_STAGE.reset(stage_token)

        return run

    return decorate


def scene_span(stage: str):
    def decorate(method):
        @wraps(method)
        def run(self, ctx, scene_id, state, *args, **kwargs):
            span_id = uuid4().hex
            fields = {
                "span_id": span_id,
                "parent_span_id": CURRENT_SPAN.get(),
                "stage": stage,
                "scene_id": scene_id,
                "scene_revision": state.revision,
            }
            stage_token = CURRENT_STAGE.set(stage)
            span_token = CURRENT_SPAN.set(span_id)
            self._emit("stage_span_started", **fields)
            started = time.monotonic()
            outcome = "completed"
            try:
                result = method(self, ctx, scene_id, state, *args, **kwargs)
                if state.failed or state.give_up:
                    outcome = "failed"
                return result
            except BaseException:
                outcome = "error"
                raise
            finally:
                self._emit(
                    "stage_span_finished",
                    **fields,
                    duration_seconds=time.monotonic() - started,
                    outcome=outcome,
                    completed_revision=state.revision,
                    submission_token=state.submission.token if state.submission is not None else "",
                )
                CURRENT_SPAN.reset(span_token)
                CURRENT_STAGE.reset(stage_token)

        return run

    return decorate
