"""Durable model request reservations and request-scoped diagnostics."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel


class RequestBudgetExceeded(RuntimeError):
    """A request was rejected before contacting a model endpoint."""


class RequestReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    call_id: str = ""
    parent_span_id: str = ""
    attempt: int = Field(default=1, ge=1)
    json_repair_attempt: int = Field(default=0, ge=0)
    scene_id: int | None = Field(default=None, ge=1)
    scene_revision: int = Field(default=0, ge=0)
    stage: str
    profile: str
    model: str
    reserved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RequestBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scene_limit: int = Field(default=64, ge=1)
    run_limit: int = Field(default=512, ge=1)
    reservations: list[RequestReservation] = Field(default_factory=list)

    def reserve(self, reservation: RequestReservation) -> None:
        if any(item.request_id == reservation.request_id for item in self.reservations):
            raise ValueError("重复的模型请求标识")
        if len(self.reservations) >= self.run_limit:
            raise RequestBudgetExceeded(f"运行模型请求预算已耗尽（{self.run_limit}）")
        if reservation.scene_id is not None:
            used = sum(item.scene_id == reservation.scene_id for item in self.reservations)
            if used >= self.scene_limit:
                raise RequestBudgetExceeded(
                    f"Scene {reservation.scene_id} 模型请求预算已耗尽（{self.scene_limit}）"
                )
        self.reservations.append(reservation)

    def integrity_errors(self) -> list[str]:
        errors = []
        if len(self.reservations) > self.run_limit:
            errors.append("模型请求记录超过运行限额")
        counts = Counter(item.scene_id for item in self.reservations if item.scene_id is not None)
        if any(count > self.scene_limit for count in counts.values()):
            errors.append("模型请求记录超过场景限额")
        if len({item.request_id for item in self.reservations}) != len(self.reservations):
            errors.append("模型请求标识重复")
        if any(item.reserved_at.utcoffset() is None for item in self.reservations):
            errors.append("模型请求时间戳缺少时区")
        return errors


@dataclass(frozen=True)
class RequestContext:
    reserve: Callable[..., RequestReservation]
    emit: Callable[[str, dict[str, Any]], None]

    def complete(self, receipt: RequestReservation, started: float, response=None, error=None):
        usage = getattr(response, "usage", None)
        tokens = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                tokens[key] = value
        self.emit(
            "model_request_finished",
            {
                **receipt.model_dump(mode="json"),
                "span_id": receipt.request_id,
                "latency_ms": (time.monotonic() - started) * 1000,
                "status": "error" if error is not None else "completed",
                "error_type": type(error).__name__ if error is not None else "",
                "usage": tokens or None,
            },
        )


class ObservedStream:
    """Observe a complete streamed attempt, including early close and iteration failures."""

    def __init__(self, stream, context: RequestContext, receipt, started):
        self.stream = stream
        self.context = context
        self.receipt = receipt
        self.started = started
        self.last_chunk = None
        self.finished = False

    def _finish(self, error=None):
        if not self.finished:
            self.finished = True
            self.context.complete(self.receipt, self.started, self.last_chunk, error)

    def __iter__(self) -> Iterator:
        try:
            for chunk in self.stream:
                if getattr(chunk, "usage", None) is not None:
                    self.last_chunk = chunk
                yield chunk
        except BaseException as exc:
            self._finish(exc)
            raise
        else:
            self._finish()

    def close(self):
        try:
            close = getattr(self.stream, "close", None)
            if callable(close):
                close()
        finally:
            self._finish(RuntimeError("stream closed before completion"))
