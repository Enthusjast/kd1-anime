"""Adapt committed pipeline events to the existing two-argument UI callback."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    name: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CallbackAdapter:
    callback: Callable[[str, dict], None] | None

    def publish(self, event: PipelineEvent) -> None:
        if self.callback is not None:
            try:
                self.callback(event.name, dict(event.data))
            except Exception:
                # UI failure must never turn a completed model/render call into
                # a retry, nor interfere with checkpoint failure handling.
                logging.getLogger(__name__).warning(
                    "Pipeline display callback failed", exc_info=False
                )


class BufferedCallback:
    """Bounded UI delivery; overflow requests a fresh committed snapshot."""

    def __init__(self, callback, *, resync=None, capacity=4096):
        if capacity < 1:
            raise ValueError("Event queue capacity must be positive")
        self.adapter = CallbackAdapter(callback)
        self.resync = resync
        self.capacity = capacity
        self.queue = deque()
        self.condition = threading.Condition()
        self.closed = False
        self.needs_resync = False
        self.thread = threading.Thread(target=self._run, name="pipeline-display", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __call__(self, name, data):
        event = PipelineEvent(name, dict(data))
        if threading.current_thread() is self.thread:
            self.adapter.publish(event)
            return
        with self.condition:
            if self.closed and threading.current_thread() is not self.thread:
                return
            if name.endswith(("progress", "stream_chunk")):
                key = (name, data.get("scene_id"))
                for index in range(len(self.queue) - 1, -1, -1):
                    old = self.queue[index]
                    if (old.name, old.data.get("scene_id")) == key:
                        self.queue[index] = event
                        return
            if len(self.queue) >= self.capacity:
                self.queue.popleft()
                self.needs_resync = True
            self.queue.append(event)
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.needs_resync or self.closed)
                if not self.queue and not self.needs_resync and self.closed:
                    return
                event = self.queue.popleft() if self.queue else None
                resync = self.needs_resync and not self.queue
                if resync:
                    self.needs_resync = False
            if event is not None:
                self.adapter.publish(event)
            if resync and self.resync is not None:
                try:
                    self.resync()
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Display snapshot refresh failed", exc_info=False
                    )

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=5)

    def __exit__(self, *args):
        self.close()
