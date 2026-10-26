"""Cooperative cancellation and logical-operation deadlines."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Event


class RunCancelled(RuntimeError):
    pass


class OperationDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class RunControl:
    cancelled: Event
    stopped: Event | None = None
    deadline: float | None = None
    clock: object = time.monotonic

    def check(self):
        if self.cancelled.is_set() or (self.stopped is not None and self.stopped.is_set()):
            raise RunCancelled("Run cancelled before completing the operation")
        if self.deadline is not None and self.clock() >= self.deadline:
            raise OperationDeadlineExceeded("Logical operation deadline exceeded")

    def remaining(self):
        self.check()
        return None if self.deadline is None else max(0.0, self.deadline - self.clock())

    def child(self, timeout):
        deadline = self.clock() + timeout if timeout > 0 else None
        if self.deadline is not None:
            deadline = min(deadline, self.deadline) if deadline is not None else self.deadline
        return RunControl(self.cancelled, self.stopped, deadline, self.clock)

    def wait(self, seconds):
        end = self.clock() + max(0.0, seconds)
        while self.clock() < end:
            remaining = self.remaining()
            self.cancelled.wait(
                min(0.1, end - self.clock(), remaining if remaining is not None else 0.1)
            )
        self.check()

    @contextmanager
    def slot(self, semaphore):
        while True:
            remaining = self.remaining()
            if semaphore.acquire(timeout=min(0.1, remaining if remaining is not None else 0.1)):
                break
        try:
            self.check()
            yield
        finally:
            semaphore.release()


CURRENT_CONTROL: ContextVar[RunControl | None] = ContextVar("kd1_run_control", default=None)


def controlled_call(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        parent = CURRENT_CONTROL.get()
        control = parent or getattr(self, "control", None)
        if control is None and self.profile.timeout_total > 0:
            control = RunControl(Event())
        if control is None:
            return method(self, *args, **kwargs)
        control = control.child(self.profile.timeout_total)
        token = CURRENT_CONTROL.set(control)
        try:
            control.check()
            result = method(self, *args, **kwargs)
            control.check()
            return result
        finally:
            CURRENT_CONTROL.reset(token)

    return call
