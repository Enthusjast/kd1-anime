"""Bounded context-preserving workers and FIFO task identities."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass

from kd1_anime.domain.dependencies import WorkIdentity


class ContextPool(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        context = copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class WorkItem:
    scene_id: int
    stage: str
    identity: WorkIdentity


class ReadyQueue:
    def __init__(self):
        self.items = deque()
        self.scenes = set()

    def push(self, item):
        if item.scene_id not in self.scenes:
            self.items.append(item)
            self.scenes.add(item.scene_id)

    def pop(self):
        item = self.items.popleft()
        self.scenes.remove(item.scene_id)
        return item
