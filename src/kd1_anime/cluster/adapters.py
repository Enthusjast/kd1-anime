"""Compatibility for older injected backends, outside scheduling policy."""

from __future__ import annotations

import inspect

from kd1_anime.cluster.jobs import PollObservation


class BackendAdapter:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def poll(self, job_ids):
        if "poll_all_statuses" in getattr(self.backend, "__dict__", {}):
            return PollObservation(
                dict(self.backend.poll_all_statuses(job_ids)),
                dict(getattr(self.backend, "last_start_times", {})),
                getattr(self.backend, "last_status_diagnostic", ""),
            )
        method = getattr(self.backend, "poll", None)
        if callable(method):
            return method(job_ids)
        statuses = self.backend.poll_all_statuses(job_ids)
        return PollObservation(
            dict(statuses),
            dict(getattr(self.backend, "last_start_times", {})),
            getattr(self.backend, "last_status_diagnostic", ""),
        )

    def submit_scene(self, *args, **kwargs):
        method = self.backend.submit_scene
        parameters = inspect.signature(method).parameters
        if "attempt_token" not in parameters and not any(
            p.kind == p.VAR_KEYWORD for p in parameters.values()
        ):
            kwargs.pop("attempt_token", None)
        return method(*args, **kwargs)

    def forward_log(self, job, positions):
        method = getattr(self.backend, "forward_log", None) or getattr(
            self.backend, "_forward_log", None
        )
        if callable(method):
            method(job, positions)

    def classify_gone(self, job):
        method = getattr(self.backend, "classify_gone", None) or getattr(
            self.backend, "_classify_gone", None
        )
        return method(job) if callable(method) else None

    def cancel_for_monitor_failure(self, job, *, status, reason):
        try:
            cancelled = self.backend.cancel_job(job.job_id)
        except Exception:
            cancelled = False
        job.cancelled = cancelled
        job.status = status if cancelled else "CANCEL_FAILED"
        job.failure_reason = reason if cancelled else reason + "; scancel 失败，禁止自动重提"


def adapt_backend(backend):
    return backend if isinstance(backend, BackendAdapter) else BackendAdapter(backend)
