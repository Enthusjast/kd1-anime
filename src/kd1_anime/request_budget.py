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
    invocation_id: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    estimated_prompt_tokens: int | None = Field(default=None, ge=0)
    reserved_output_tokens: int | None = Field(default=None, ge=0)
    reserved_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input_price_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_price_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reserved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RequestBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scene_limit: int = Field(default=64, ge=1)
    run_limit: int = Field(default=512, ge=1)
    token_limit: int | None = Field(default=None, ge=1)
    cost_limit_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    reservations: list[RequestReservation] = Field(default_factory=list)

    @staticmethod
    def _valid_cost(item: RequestReservation) -> bool:
        values = (
            item.estimated_prompt_tokens,
            item.reserved_output_tokens,
            item.input_price_usd_per_million,
            item.output_price_usd_per_million,
            item.reserved_cost_usd,
        )
        if any(value is None for value in values):
            return False
        expected = (
            item.estimated_prompt_tokens * item.input_price_usd_per_million
            + item.reserved_output_tokens * item.output_price_usd_per_million
        ) / 1_000_000
        return item.reserved_cost_usd >= expected

    def check_reported_usage(self, receipts) -> None:
        if self.token_limit is None and self.cost_limit_usd is None:
            return
        reservations = {item.request_id: item for item in self.reservations}
        for usage in receipts:
            if usage.usage_source != "provider_reported":
                continue
            item = reservations.get(usage.request_id)
            if item is None:
                raise RequestBudgetExceeded("Provider usage lacks its reservation")
            if (
                (
                    usage.prompt_tokens is not None
                    and item.estimated_prompt_tokens is not None
                    and usage.prompt_tokens > item.estimated_prompt_tokens
                )
                or (
                    usage.completion_tokens is not None
                    and item.reserved_output_tokens is not None
                    and usage.completion_tokens > item.reserved_output_tokens
                )
                or (
                    usage.total_tokens is not None
                    and item.estimated_prompt_tokens is not None
                    and item.reserved_output_tokens is not None
                    and usage.total_tokens
                    > item.estimated_prompt_tokens + item.reserved_output_tokens
                )
            ):
                raise RequestBudgetExceeded("Provider usage exceeded its reserved request envelope")

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
        records = (*self.reservations, reservation)
        if self.token_limit is not None:
            if any(
                item.estimated_prompt_tokens is None or item.reserved_output_tokens is None
                for item in records
            ):
                raise RequestBudgetExceeded(
                    "Token budget requires a known bounded request envelope"
                )
            if (
                sum(item.estimated_prompt_tokens + item.reserved_output_tokens for item in records)
                > self.token_limit
            ):
                raise RequestBudgetExceeded("Run token budget exhausted")
        if self.cost_limit_usd is not None:
            if any(not self._valid_cost(item) for item in records):
                raise RequestBudgetExceeded(
                    "Hard cost budget requires known model prices and bounded output"
                )
            if sum(item.reserved_cost_usd for item in records) > self.cost_limit_usd:
                raise RequestBudgetExceeded("Run cost budget exhausted")
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
        if self.token_limit is not None and (
            any(
                item.estimated_prompt_tokens is None or item.reserved_output_tokens is None
                for item in self.reservations
            )
            or sum(
                item.estimated_prompt_tokens + item.reserved_output_tokens
                for item in self.reservations
                if item.estimated_prompt_tokens is not None
                and item.reserved_output_tokens is not None
            )
            > self.token_limit
        ):
            errors.append("Token reservations exceed or lack the pinned limit")
        if self.cost_limit_usd is not None and (
            any(not self._valid_cost(item) for item in self.reservations)
            or sum(item.reserved_cost_usd or 0 for item in self.reservations) > self.cost_limit_usd
        ):
            errors.append("Cost reservations exceed or lack the pinned limit")
        return errors


@dataclass(frozen=True)
class RequestContext:
    reserve: Callable[..., RequestReservation]
    emit: Callable[[str, dict[str, Any]], None]

    def complete(
        self,
        receipt: RequestReservation,
        started: float,
        response=None,
        error=None,
        *,
        estimated_completion_tokens=None,
    ):
        usage = getattr(response, "usage", None)
        tokens = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                tokens[key] = value
        if (
            not tokens
            and error is None
            and estimated_completion_tokens is None
            and response is not None
        ):
            choices = getattr(response, "choices", None) or ()
            text = (
                getattr(getattr(choices[0], "message", None), "content", None) if choices else None
            )
            if isinstance(text, str):
                estimated_completion_tokens = len(text.encode("utf-8"))
        estimated = None
        if (
            not tokens
            and error is None
            and estimated_completion_tokens is not None
            and receipt.estimated_prompt_tokens is not None
        ):
            estimated = {
                "prompt_tokens": receipt.estimated_prompt_tokens,
                "completion_tokens": estimated_completion_tokens,
                "total_tokens": receipt.estimated_prompt_tokens + estimated_completion_tokens,
            }
        self.emit(
            "model_request_finished",
            {
                **receipt.model_dump(mode="json"),
                "span_id": receipt.request_id,
                "latency_ms": (time.monotonic() - started) * 1000,
                "status": "error" if error is not None else "completed",
                "error_type": type(error).__name__ if error is not None else "",
                "usage": tokens or None,
                "estimated_usage": estimated,
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
        self.visible_bytes = 0
        self.finished = False

    def _finish(self, error=None):
        if not self.finished:
            self.finished = True
            self.context.complete(
                self.receipt,
                self.started,
                self.last_chunk,
                error,
                estimated_completion_tokens=self.visible_bytes if error is None else None,
            )

    def __iter__(self) -> Iterator:
        try:
            for chunk in self.stream:
                if getattr(chunk, "usage", None) is not None:
                    self.last_chunk = chunk
                choices = getattr(chunk, "choices", None) or ()
                text = (
                    getattr(getattr(choices[0], "delta", None), "content", None)
                    if choices
                    else None
                )
                if isinstance(text, str):
                    self.visible_bytes += len(text.encode("utf-8"))
                yield chunk
        except BaseException as exc:
            try:
                close = getattr(self.stream, "close", None)
                if callable(close):
                    close()
            finally:
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
