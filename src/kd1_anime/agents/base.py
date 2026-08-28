"""
BaseAgent 基类
封装 OpenAI 兼容 API 调用,提供:
- 指数退避重试机制
- 结构化 JSON 输出 (支持 Pydantic 模型校验)
- Rich 控制台美化输出 (Agent 思考过程)
- 健壮的 JSON / 代码块提取 (容错散文包裹、fence、截断)
"""

import json
import random
import re
import time
from contextlib import suppress
from typing import ClassVar, Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel

from kd1_anime.config import LLMRuntimeProfile, settings
from kd1_anime.llm_cache import LLMResponseCache, make_cache_key

T = TypeVar("T", bound=BaseModel)

console = Console()


class StreamCancelledError(RuntimeError):
    """用户主动取消流式响应。"""


class TruncatedResponseError(RuntimeError):
    """模型反复达到输出上限，响应不能安全消费。"""


class BaseAgent:
    """所有 Agent 的基类,封装 LLM 交互逻辑"""

    name: str = "BaseAgent"

    def __init__(self, profile: LLMRuntimeProfile | None = None):
        # 延迟构造 client: 避免 API Key 为空时在实例化阶段就崩溃
        # (让调用方有机会先给出友好的配置错误提示)
        self.profile = profile or settings.main_llm_profile()
        self._client: OpenAI | None = None
        self.model = self.profile.model
        self.last_call_metrics: dict[str, object] = {}
        self._last_usage: dict[str, int] = {}

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI client"""
        if self._client is None:
            import httpx

            self._client = OpenAI(
                api_key=self.profile.api_key,
                base_url=self.profile.base_url,
                timeout=httpx.Timeout(
                    self.profile.timeout_read,
                    connect=self.profile.timeout_connect,
                    read=self.profile.timeout_read,
                ),
            )
        return self._client

    def check_api_available(
        self,
        *,
        timeout: float | None = None,
        messages: list[dict] | None = None,
    ) -> None:
        """发送一次最小聊天请求，确认当前端点和模型确实可调用。

        只检查连接、鉴权和模型路由，不消费业务提示词。请求使用一个 token
        上限，且禁用 SDK 重试，避免 CLI 启动时因端点故障长时间等待或重复计费。
        某些推理模型/兼容端点不接受 ``max_tokens``，会再尝试一次不带该参数
        的请求；其他错误直接交给调用方转换为启动失败。
        """

        self.profile.require()
        health_timeout = self.profile.healthcheck_timeout if timeout is None else float(timeout)
        request = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": "Reply with OK."}],
        }
        if self.profile.send_max_tokens:
            request["max_tokens"] = 1

        try:
            client = self.client
            with_options = getattr(client, "with_options", None)
            if callable(with_options):
                client = with_options(timeout=health_timeout, max_retries=0)
            response = client.chat.completions.create(**request)
        except BadRequestError as exc:
            # 兼容只支持 max_completion_tokens 或完全忽略 token 上限的端点。
            if "max_tokens" not in request or "max_tokens" not in str(exc).lower():
                raise RuntimeError(self._healthcheck_error(exc)) from exc
            request.pop("max_tokens")
            try:
                response = client.chat.completions.create(**request)
            except Exception as retry_error:
                raise RuntimeError(self._healthcheck_error(retry_error)) from retry_error
        except Exception as exc:
            raise RuntimeError(self._healthcheck_error(exc)) from exc

        if not getattr(response, "choices", None):
            raise RuntimeError(self._healthcheck_error("API 返回了空 choices"))

    def _healthcheck_error(self, error: object) -> str:
        """生成不泄露 API Key 的启动探测错误。"""

        detail = str(error).strip() or type(error).__name__
        if self.profile.api_key:
            detail = detail.replace(self.profile.api_key, "<redacted>")
        return (
            f"{self.profile.label}API 不可用"
            f"（{self._safe_endpoint(self.profile.base_url)}，模型 {self.model}）：{detail}"
        )

    @staticmethod
    def _safe_endpoint(base_url: str) -> str:
        """只显示 scheme/host/path，避免 URL 查询参数或密码泄露。"""

        try:
            parsed = urlsplit(base_url)
            host = parsed.hostname or "<invalid-host>"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            return "<invalid-endpoint>"

    @staticmethod
    def _extract_usage(usage: object) -> dict[str, int]:
        if usage is None:
            return {}
        values: dict[str, int] = {}
        for target, names in (
            ("prompt_tokens", ("prompt_tokens", "input_tokens")),
            ("completion_tokens", ("completion_tokens", "output_tokens")),
        ):
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if isinstance(value, int) and value >= 0:
                    values[target] = value
                    break
        return values

    def _log(self, message: str, style: str = "bold cyan") -> None:
        """打印 Agent 思考过程（仪表盘激活时抑制，避免破坏 Live 渲染）"""
        # 延迟导入避免循环依赖
        from kd1_anime.dashboard import suppress_agent_logs

        if suppress_agent_logs():
            return
        safe_msg = str(message).replace("[", "\\[")
        safe_name = str(self.name).replace("[", "\\[")
        console.print(f"[{style}]{safe_name}[/] {safe_msg}")

    def _log_panel(self, title: str, content: str, style: str = "blue") -> None:
        """用 Panel 展示详细内容 (仪表盘激活时抑制, 避免破坏 Live 渲染)"""
        # 延迟导入避免循环依赖
        from kd1_anime.dashboard import suppress_agent_logs

        if suppress_agent_logs():
            return
        console.print(Panel(content, title=title, border_style=style))

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def call_llm(
        self,
        system_prompt: str = "",
        user_message: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        messages: list[dict] | None = None,
        stream: bool = False,
        allow_truncated: bool = False,
        cache_namespace: str = "",
    ) -> str:
        """
        调用 LLM API,内置指数退避重试

        Args:
            system_prompt: 系统提示词 (当 messages 未提供时使用)
            user_message: 用户消息 (当 messages 未提供时使用)
            temperature: 温度参数,默认使用配置值
            max_tokens: 最大 token 数,默认使用配置值
            json_mode: 是否要求 JSON 格式输出
            messages: 完整的消息列表,用于多轮对话. 提供时忽略 system_prompt 和 user_message
            stream: 是否流式输出 (边生成边打印, 避免长调用时界面冻结)
            allow_truncated: 允许调用方自行校验 ``finish_reason=length`` 的非空内容。
                仅适合有严格结构校验的响应，普通文本/代码默认拒绝截断结果。

        Returns:
            LLM 的文本响应

        Raises:
            RuntimeError: 重试耗尽后抛出
        """
        self.profile.require()

        temp = temperature if temperature is not None else self.profile.temperature
        tokens = max_tokens if max_tokens is not None else self.profile.max_tokens

        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        # 只对调用方明确要求的非流式请求启用缓存。交互式流式输出包含
        # 用户取消、实时显示等语义，不能被缓存；静默流式传输仍属于可缓存的
        # 非流式业务调用。缓存命中直接跳过网络和重试，不会绕过 profile.require。
        cache: LLMResponseCache | None = None
        cache_key: str | None = None
        started_at = time.monotonic()
        uses_default_client = getattr(type(self), "client", None) is BaseAgent.client
        if not stream and settings.LLM_CACHE_ENABLED and uses_default_client:
            cache = LLMResponseCache()
            cache_key = make_cache_key(
                self.profile,
                messages,
                temperature=temp,
                max_tokens=tokens,
                json_mode=json_mode,
                allow_truncated=allow_truncated,
                extra=(
                    f"{self.name}:{type(self).__module__}.{type(self).__qualname__}:"
                    f"{cache_namespace}"
                ),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                cache.record_call(
                    cache_key,
                    cache_hit=True,
                    latency_ms=0.0,
                    model=self.model,
                )
                self.last_call_metrics = {
                    "cache_hit": True,
                    "latency_ms": 0.0,
                    "attempts": 0,
                    "model": self.model,
                }
                return cached

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        # 部分 (推理类) 模型拒绝 max_tokens; 通过设置可关闭
        if self.profile.send_max_tokens and tokens:
            kwargs["max_tokens"] = tokens
        if json_mode and self.profile.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        json_fallback_used = False
        stream_fallback_used = False
        # 静默流式: stream=False(不展示内容)时仍使用流式传输, 避免长生成超时
        use_stream_transport = stream or (not stream and self.profile.silent_stream)
        temp_fallback_used = False
        max_tokens_fallback_used = False
        max_tokens_boosted = False

        attempt = 0
        while attempt < self.profile.max_retries:
            attempt += 1
            try:
                self._log(f"LLM 调用中... (尝试 {attempt}/{self.profile.max_retries})")
                if self.profile.debug:
                    for i, msg in enumerate(messages):
                        role = msg["role"]
                        body = msg["content"]
                        if isinstance(body, str):
                            preview = body[:500] + ("..." if len(body) > 500 else "")
                        else:
                            preview = ", ".join(
                                str(item.get("type", "unknown"))
                                for item in body
                                if isinstance(item, dict)
                            )
                        console.print(f"[dim]DEBUG [{role} #{i}]: {preview}[/]", markup=False)
                self._last_usage = {}
                if use_stream_transport:
                    content, finish_reason = self._stream_llm(kwargs, display=stream)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    self._last_usage = self._extract_usage(getattr(response, "usage", None))
                    content = content.strip()
                if not content:
                    finish = finish_reason or "stream_empty"
                    if finish == "length":
                        # 空内容 + length: 推理模型常把输出预算耗尽在 reasoning 上,
                        # 导致 content 为空且 finish=length。先补足 max_tokens 重试
                        # (不消耗业务重试), 而不是立刻把整个场景判死。
                        if not max_tokens_boosted and not max_tokens_fallback_used:
                            max_tokens_val = kwargs.get("max_tokens")
                            boost = self.profile.empty_retry_max_tokens
                            if max_tokens_val:
                                boost = max(int(max_tokens_val) * 2, boost)
                            boost = min(boost, 65536)
                            kwargs["max_tokens"] = boost
                            max_tokens_boosted = True
                            self._log(
                                f"空响应(length): 补充 max_tokens={boost} 后重试 "
                                "(推理模型可能耗尽输出预算)",
                                style="yellow",
                            )
                            attempt -= 1  # 参数修复, 不消耗业务重试次数
                            continue
                        # 已补足过 max_tokens 仍为空 → 走正常重试, 重试耗尽后才报错
                        self._log(
                            "LLM 返回空响应(length), 将重试... "
                            f"(max_tokens={kwargs.get('max_tokens')})",
                            style="bold yellow",
                        )
                        last_error = TruncatedResponseError("LLM 输出被截断且内容为空")
                        if attempt < self.profile.max_retries:
                            delay = self._retry_delay(attempt, last_error)
                            time.sleep(delay)
                        continue
                    if json_mode and "response_format" in kwargs and not json_fallback_used:
                        self._log(
                            "端点在 response_format 模式返回空内容，降级为 prompt-only JSON 重试",
                            style="yellow",
                        )
                        kwargs.pop("response_format", None)
                        json_fallback_used = True
                        # 在系统提示中添加明确的 JSON 输出要求
                        for msg in kwargs.get("messages", []):
                            if msg.get("role") == "system":
                                msg["content"] += (
                                    "\n\n重要：你必须返回有效的 JSON 格式。不要包含任何其他文本，只返回 JSON。"
                                )
                                break
                        attempt -= 1  # 参数兼容性降级，不消耗业务重试次数
                        continue
                    if not stream and not use_stream_transport and not stream_fallback_used:
                        self._log(
                            "端点非流式响应为空，切换为静默流式传输重试",
                            style="yellow",
                        )
                        use_stream_transport = True
                        stream_fallback_used = True
                        attempt -= 1  # 传输兼容性降级，不消耗业务重试次数
                        continue
                    # 推理模型空响应：常因 reasoning_content 耗尽服务端默认输出上限。
                    # 补上充足 max_tokens 后重试, 避免反复拿到空响应。
                    if (
                        "max_tokens" not in kwargs
                        and not max_tokens_boosted
                        and not max_tokens_fallback_used
                    ):
                        boost = self.profile.empty_retry_max_tokens
                        self._log(
                            f"空响应: 补充 max_tokens={boost} 后重试 (推理模型可能耗尽输出预算)",
                            style="yellow",
                        )
                        kwargs["max_tokens"] = boost
                        max_tokens_boosted = True
                        attempt -= 1  # 参数修复, 不消耗业务重试次数
                        continue
                    self._log(
                        f"LLM 返回空响应, 将重试... (finish_reason={finish})",
                        style="bold yellow",
                    )
                    last_error = RuntimeError("LLM 返回空响应")
                    if attempt < self.profile.max_retries:
                        delay = self._retry_delay(attempt, last_error)
                        time.sleep(delay)
                    continue
                if finish_reason == "length":
                    if allow_truncated:
                        if cache is not None and cache_key is not None:
                            cache.set(
                                cache_key,
                                content,
                                latency_ms=(time.monotonic() - started_at) * 1000,
                                prompt_tokens=self._last_usage.get("prompt_tokens"),
                                completion_tokens=self._last_usage.get("completion_tokens"),
                            )
                            cache.record_call(
                                cache_key,
                                cache_hit=False,
                                latency_ms=(time.monotonic() - started_at) * 1000,
                                prompt_tokens=self._last_usage.get("prompt_tokens"),
                                completion_tokens=self._last_usage.get("completion_tokens"),
                                model=self.model,
                            )
                        self.last_call_metrics = {
                            "cache_hit": False,
                            "latency_ms": (time.monotonic() - started_at) * 1000,
                            "attempts": attempt,
                            "model": self.model,
                            **self._last_usage,
                        }
                        return content
                    if not max_tokens_boosted and not max_tokens_fallback_used:
                        current_limit = kwargs.get("max_tokens")
                        boost = self.profile.empty_retry_max_tokens
                        if current_limit:
                            boost = max(int(current_limit) * 2, boost)
                        boost = min(boost, 65536)
                        kwargs["max_tokens"] = boost
                        max_tokens_boosted = True
                        self._log(
                            f"LLM 响应被截断: 提高 max_tokens={boost} 后重新生成",
                            style="yellow",
                        )
                        attempt -= 1
                        continue
                    last_error = TruncatedResponseError("LLM 输出被截断")
                    self._log("LLM 响应仍被截断，将重新生成", style="bold yellow")
                    if attempt < self.profile.max_retries:
                        time.sleep(self._retry_delay(attempt, last_error))
                    continue
                if self.profile.debug and (not use_stream_transport or not stream):
                    preview = content[:500] + ("..." if len(content) > 500 else "")
                    console.print(f"[dim]DEBUG [response]: {preview}[/]", markup=False)
                if cache is not None and cache_key is not None:
                    cache.set(
                        cache_key,
                        content,
                        latency_ms=(time.monotonic() - started_at) * 1000,
                        prompt_tokens=self._last_usage.get("prompt_tokens"),
                        completion_tokens=self._last_usage.get("completion_tokens"),
                    )
                    cache.record_call(
                        cache_key,
                        cache_hit=False,
                        latency_ms=(time.monotonic() - started_at) * 1000,
                        prompt_tokens=self._last_usage.get("prompt_tokens"),
                        completion_tokens=self._last_usage.get("completion_tokens"),
                        model=self.model,
                    )
                self.last_call_metrics = {
                    "cache_hit": False,
                    "latency_ms": (time.monotonic() - started_at) * 1000,
                    "attempts": attempt,
                    "model": self.model,
                    **self._last_usage,
                }
                return content

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                # 关闭 client 重建, 避免 httpx 连接池复用死连接
                self._client = None
                if attempt < self.profile.max_retries:
                    delay = self._retry_delay(attempt, e)
                    self._log(
                        f"API 限流/超时/连接错误, {delay:.1f}s 后重试... ({e})",
                        style="bold yellow",
                    )
                    time.sleep(delay)

            except StreamCancelledError:
                raise

            except BadRequestError as e:
                # 400 通常是确定性错误,不应重试. 但若由 response_format 引起,
                # 降级为"只用 prompt 要求 JSON"再试一次 (不消耗额外重试预算外的一次).
                msg = str(e).lower()
                if (
                    json_mode
                    and not json_fallback_used
                    and ("response_format" in msg or "json" in msg or "format" in msg)
                ):
                    self._log(
                        "端点不支持 response_format, 降级为 prompt-only JSON 重试", style="yellow"
                    )
                    kwargs.pop("response_format", None)
                    json_fallback_used = True
                    # 在系统提示中添加明确的 JSON 输出要求
                    for msg_item in kwargs.get("messages", []):
                        if msg_item.get("role") == "system":
                            msg_item["content"] += (
                                "\n\n重要：你必须返回有效的 JSON 格式。不要包含任何其他文本，只返回 JSON。"
                            )
                            break
                    attempt -= 1  # 参数修复, 不消耗重试次数
                    continue
                # Kimi 等推理模型只允许 temperature=1
                if not temp_fallback_used and ("temperature" in msg or "only 1 is allowed" in msg):
                    self._log("模型仅支持 temperature=1, 降级重试", style="yellow")
                    kwargs["temperature"] = 1.0
                    temp_fallback_used = True
                    attempt -= 1  # 参数修复, 不消耗重试次数
                    continue
                # 模型拒绝 max_tokens 参数
                if not max_tokens_fallback_used and "max_tokens" in msg and "max_tokens" in kwargs:
                    self._log("模型不支持 max_tokens 参数, 移除后重试", style="yellow")
                    kwargs.pop("max_tokens", None)
                    max_tokens_fallback_used = True
                    attempt -= 1  # 参数修复, 不消耗重试次数
                    continue
                # 其他 400 直接失败, 不重试
                raise RuntimeError(f"[{self.name}] 请求被拒绝 (400): {e}") from e

            except APIError as e:
                # 打印完整错误详情 (含 HTTP 状态码和响应体)
                self._log(
                    "API 错误 "
                    f"(status={getattr(e, 'status_code', '?')}, "
                    f"type={getattr(e, 'type', type(e).__name__)}): "
                    f"{getattr(e, 'message', str(e))}",
                    style="bold yellow",
                )
                # 请求被拒绝 / 被封 — 不可重试
                blocked_keywords = ["blocked", "rejected", "denied", "forbidden"]
                msg_lower = str(e).lower()
                if any(kw in msg_lower for kw in blocked_keywords):
                    raise RuntimeError(f"[{self.name}] API 请求被拒绝 (不可重试): {e}") from e
                last_error = e
                if attempt < self.profile.max_retries:
                    delay = self._retry_delay(attempt, e)
                    time.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"[{self.name}] LLM 调用发生未知错误: {e}") from e

        if isinstance(last_error, TruncatedResponseError):
            raise TruncatedResponseError(f"[{self.name}] LLM 输出在重试后仍被截断") from last_error
        raise RuntimeError(
            f"[{self.name}] LLM 调用在 {self.profile.max_retries} 次重试后仍然失败: {last_error}"
        )

    def _retry_delay(self, attempt: int, error: Exception | None = None) -> float:
        """指数退避加随机抖动，并优先尊重服务端 Retry-After。"""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(300.0, max(0.1, float(retry_after)))
            except (TypeError, ValueError):
                pass
        base = self.profile.retry_base_delay * (2 ** max(0, attempt - 1))
        return min(300.0, base + random.uniform(0, self.profile.retry_base_delay))

    def _stream_llm(self, kwargs: dict, *, display: bool = True) -> tuple[str, str | None]:
        """流式调用 LLM；可静默收集，显示时 ESC 可取消。

        Returns:
            (content, finish_reason): finish_reason 为流中最后出现的值，
            用于空响应时区分 length/stop/stream_empty。
        """
        import select
        import sys
        import threading

        kwargs = {**kwargs, "stream": True}
        chunks: list[str] = []
        reasoning_chunks = 0
        content_chunks = 0
        empty_chunks = 0
        cancelled = threading.Event()
        last_finish: str | None = None
        self._last_usage = {}

        def _watch_esc() -> None:
            try:
                while not cancelled.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if not readable:
                        continue
                    byte = sys.stdin.buffer.read(1)
                    if not byte:
                        break
                    if byte == b"\x1b":
                        # ESC 是转义序列起始字节（方向键 ^[[D、鼠标事件等）。
                        # 等待一小段窗口：若后面紧跟其他字节则是转义序列（忽略），
                        # 只有"独立 ESC"（用户按 ESC）才取消流式输出。
                        r2, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if not r2:
                            # 没有后续字节 → 独立 ESC，用户取消
                            cancelled.set()
                            break
                        # 有后续字节 → 转义序列，丢弃它
                        with suppress(OSError):
                            sys.stdin.buffer.read1(8)
            except (OSError, ValueError):
                return

        watcher = None
        if display and threading.current_thread() is threading.main_thread() and sys.stdin.isatty():
            watcher = threading.Thread(target=_watch_esc, daemon=True)
            watcher.start()

        stream = None
        user_cancelled = False
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                chunk_usage = self._extract_usage(getattr(chunk, "usage", None))
                if chunk_usage:
                    self._last_usage.update(chunk_usage)
                if cancelled.is_set():
                    user_cancelled = True
                    if display:
                        console.print("\n[dim](ESC 取消)[/]")
                    break

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning_content", None)
                finish = getattr(chunk.choices[0], "finish_reason", None)
                if finish:
                    last_finish = finish
                if reasoning:
                    if display and reasoning_chunks == 0:
                        console.print("[dim]思考:[/] ", end="")
                    reasoning_chunks += 1
                    if display and reasoning_chunks % 20 == 0:
                        console.print("[dim].[/]", end="")
                if text:
                    content_chunks += 1
                    chunks.append(text)
                    if display:
                        console.print(text, end="", soft_wrap=True, highlight=False)
                elif not reasoning and not finish:
                    empty_chunks += 1
                    if display and empty_chunks <= 3 and self.profile.debug:
                        delta_fields = {
                            key: value for key, value in delta.__dict__.items() if value is not None
                        }
                        console.print(
                            f"\n[dim]DEBUG 空 chunk delta: {delta_fields}[/]", markup=False
                        )
                elif finish:
                    if display:
                        now = time.strftime("%H:%M:%S")
                        console.print(
                            f"\n[dim]({now} 流结束: {finish}, "
                            f"思考 {reasoning_chunks}, 内容 {content_chunks}, 空 {empty_chunks})[/]"
                        )
        finally:
            cancelled.set()
            if stream is not None:
                with suppress(AttributeError, OSError, RuntimeError):
                    stream.close()
            if watcher is not None:
                watcher.join(timeout=0.5)

        if user_cancelled:
            raise StreamCancelledError(f"[{self.name}] 用户取消了流式响应")
        if display and content_chunks == 0 and self.profile.debug:
            console.print(
                f"[bold red]诊断:[/] 无 content. reasoning={reasoning_chunks}, "
                "可能是推理模型耗尽 token 或 API 异常."
            )
        if display and reasoning_chunks > 0 and content_chunks == 0:
            console.print(
                "[bold yellow]警告:[/] 推理消耗了所有 token, 未生成内容. "
                "请大幅增大 LLM_MAX_TOKENS (建议 16384+)."
            )
        if display:
            console.print()
        return "".join(chunks).strip(), last_finish

    def call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        temperature: float | None = None,
        stream: bool = False,
        *,
        messages: list[dict] | None = None,
        max_tokens: int | None = None,
        allow_truncated: bool = False,
    ) -> T:
        """
        调用 LLM 并将响应解析为 Pydantic 模型

        输出未通过 JSON / Pydantic 结构校验时, 会带错误反馈重试
        (次数由 LLM_JSON_REPAIR_ATTEMPTS 控制), 避免一次输出不合规
        (如枚举值写错、缺字段) 就杀死整个场景。

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            response_model: 期望的 Pydantic 模型类型
            temperature: 温度参数 (默认 0.0)
            stream: 是否使用流式传输 (部分 API 非流式可能超时)
        """
        temp = 0.0 if temperature is None else temperature

        repair_attempts = max(0, self.profile.json_repair_attempts)
        current_message = user_message
        current_messages = self._clone_messages(messages) if messages is not None else None

        for attempt in range(repair_attempts + 1):
            call_kwargs = {
                "system_prompt": system_prompt,
                "user_message": current_message,
                "temperature": temp,
                "max_tokens": max_tokens,
                "json_mode": True,
                "messages": self._clone_messages(current_messages),
            }
            if getattr(type(self), "call_llm", None) is BaseAgent.call_llm:
                call_kwargs["cache_namespace"] = (
                    f"{response_model.__module__}.{response_model.__qualname__}"
                )
            if allow_truncated:
                call_kwargs["allow_truncated"] = True
            raw = self.call_llm(
                **call_kwargs,
                stream=stream,
            )

            json_str = self._extract_json(raw, expected_type="object")
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as first_error:
                repaired = self._escape_unescaped_quotes_in_json(json_str)
                repaired = self._escape_control_chars_in_json(repaired)
                repaired = self._fix_latex_escapes_in_json(repaired)
                repaired = self._close_truncated_json(repaired)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as error:
                    if attempt >= repair_attempts:
                        self._log_panel(
                            "JSON 解析失败",
                            f"原始响应:\n{raw}\n\n修复后:\n{repaired}\n\n错误: {error}",
                            style="red",
                        )
                        raise RuntimeError(
                            f"[{self.name}] LLM 返回了无效的 JSON: {first_error}"
                        ) from error
                    hint = f"上一次输出无法解析为 JSON:\n{raw[-2000:]}"
                    if current_messages is None:
                        current_message = self._append_repair_hint(current_message, hint)
                    else:
                        current_messages = self._append_messages_repair_hint(current_messages, hint)
                    self._log(
                        f"JSON 解析失败, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                        style="yellow",
                    )
                    continue

            # 修正常见拼写错误
            if isinstance(data, dict):
                typo_map = {
                    "key_momens": "key_moments",
                    "key_moment": "key_moments",
                    "visual_desgin": "visual_design",
                    "camera_movment": "camera_movement",
                    "visual_flwo": "visual_flow",
                    "computaiton": "computation",
                }
                data = {typo_map.get(k, k): v for k, v in data.items()}

            try:
                return response_model.model_validate(data)
            except ValidationError as e:
                if attempt >= repair_attempts:
                    self._log_panel(
                        "Pydantic 校验失败",
                        f"JSON 数据:\n{json.dumps(data, ensure_ascii=False, indent=2)}\n\n错误: {e}",
                        style="red",
                    )
                    raise RuntimeError(f"[{self.name}] LLM 输出不符合预期结构: {e}") from e
                hint = (
                    "上一次输出未通过结构校验:\n"
                    f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n校验错误: {e}"
                )
                if current_messages is None:
                    current_message = self._append_repair_hint(current_message, hint)
                else:
                    current_messages = self._append_messages_repair_hint(current_messages, hint)
                self._log(
                    f"输出结构不合规, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                    style="yellow",
                )

        # 理论不可达 (循环内要么 return 要么 raise)
        raise RuntimeError(f"[{self.name}] LLM 输出不符合预期结构")

    def call_llm_json_list(
        self,
        system_prompt: str,
        user_message: str,
        item_model: type[T],
        temperature: float | None = None,
        *,
        allow_truncated: bool = False,
    ) -> list[T]:
        """
        调用 LLM 并将响应解析为 Pydantic 模型列表

        期望 LLM 返回 {"items": [...]} 或直接返回 [...]
        输出未通过结构校验时带错误反馈重试 (LLM_JSON_REPAIR_ATTEMPTS)。
        """
        temp = 0.0 if temperature is None else temperature

        repair_attempts = max(0, self.profile.json_repair_attempts)
        current_message = user_message

        for attempt in range(repair_attempts + 1):
            call_kwargs = {
                "system_prompt": system_prompt,
                "user_message": current_message,
                "temperature": temp,
                "json_mode": True,
            }
            if getattr(type(self), "call_llm", None) is BaseAgent.call_llm:
                call_kwargs["cache_namespace"] = (
                    f"{item_model.__module__}.{item_model.__qualname__}"
                )
            if allow_truncated:
                call_kwargs["allow_truncated"] = True
            raw = self.call_llm(**call_kwargs)

            json_str = self._extract_json(raw, expected_type="array")
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as first_error:
                repaired = self._escape_unescaped_quotes_in_json(json_str)
                repaired = self._escape_control_chars_in_json(repaired)
                repaired = self._fix_latex_escapes_in_json(repaired)
                repaired = self._close_truncated_json(repaired)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as error:
                    if attempt >= repair_attempts:
                        raise RuntimeError(
                            f"[{self.name}] LLM 返回了无效的 JSON: {first_error}"
                        ) from error
                    current_message = self._append_repair_hint(
                        current_message,
                        f"上一次输出无法解析为 JSON:\n{raw[-2000:]}",
                    )
                    self._log(
                        f"JSON 解析失败, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                        style="yellow",
                    )
                    continue

            # 支持 {"items": [...]} 或直接 [...]
            if isinstance(data, dict) and "items" in data:
                items_data = data["items"]
            elif isinstance(data, list):
                items_data = data
            else:
                if attempt >= repair_attempts:
                    raise RuntimeError(
                        f"[{self.name}] 期望 JSON 列表或 {{'items': [...]}}, 收到: {type(data)}"
                    )
                current_message = self._append_repair_hint(
                    current_message,
                    f"上一次输出不是 JSON 数组, 收到: {type(data).__name__}",
                )
                self._log(
                    f"输出结构不合规, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                    style="yellow",
                )
                continue

            if not isinstance(items_data, list):
                if attempt >= repair_attempts:
                    raise RuntimeError(f"[{self.name}] items 必须是 JSON 数组")
                current_message = self._append_repair_hint(
                    current_message,
                    "上一次输出的 items 字段不是 JSON 数组",
                )
                self._log(
                    f"输出结构不合规, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                    style="yellow",
                )
                continue

            results = []
            validation_errors: list[str] = []
            for i, item in enumerate(items_data):
                try:
                    results.append(item_model.model_validate(item))
                except ValidationError as e:
                    validation_errors.append(f"第 {i} 项: {e}")

            if validation_errors:
                preview = "\n".join(validation_errors[:5])
                if attempt >= repair_attempts:
                    raise RuntimeError(
                        f"[{self.name}] 列表中有 {len(validation_errors)} 项未通过结构校验，"
                        f"拒绝使用残缺结果：\n{preview}"
                    )
                current_message = self._append_repair_hint(
                    current_message,
                    f"上一次输出列表中有 {len(validation_errors)} 项未通过校验:\n{preview}",
                )
                self._log(
                    f"列表项结构不合规, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                    style="yellow",
                )
                continue

            if not results:
                if attempt >= repair_attempts:
                    raise RuntimeError(f"[{self.name}] 没有任何有效的列表项通过校验")
                current_message = self._append_repair_hint(
                    current_message,
                    "上一次输出中没有有效的列表项",
                )
                self._log(
                    f"列表为空, 带错误反馈重试 ({attempt + 1}/{repair_attempts})",
                    style="yellow",
                )
                continue

            return results

        # 理论不可达
        raise RuntimeError(f"[{self.name}] 列表解析失败")

    @staticmethod
    def _append_repair_hint(user_message: str, hint: str) -> str:
        """把上一次的结构校验错误追加到用户消息, 引导模型修正后重试。"""
        return (
            f"{user_message}\n\n"
            "## 上一次输出未通过结构校验, 请修正后重新输出\n"
            f"{hint}\n\n"
            "请严格按照要求的 JSON schema 重新输出: 枚举字段必须使用给定取值之一, "
            "不得缺失必填字段, 不要包含额外字段, 只输出 JSON 本身。"
        )

    @staticmethod
    def _clone_messages(messages: list[dict] | None) -> list[dict] | None:
        """复制消息容器，避免兼容性降级逻辑改写调用方的多模态消息。"""

        if messages is None:
            return None
        cloned: list[dict] = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                item["content"] = [
                    dict(part) if isinstance(part, dict) else part for part in content
                ]
            cloned.append(item)
        return cloned

    @classmethod
    def _append_messages_repair_hint(cls, messages: list[dict], hint: str) -> list[dict]:
        """向多模态 user 消息追加纯文本 schema 修复提示，保留原图片。"""

        cloned = cls._clone_messages(messages) or []
        repair_text = cls._append_repair_hint("", hint).strip()
        for message in reversed(cloned):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [*content, {"type": "text", "text": repair_text}]
            else:
                message["content"] = f"{content or ''}\n\n{repair_text}".strip()
            return cloned
        cloned.append({"role": "user", "content": repair_text})
        return cloned

    # ------------------------------------------------------------------
    # 提取工具 (健壮版)
    # ------------------------------------------------------------------

    # JSON 合法转义字符 (RFC 8259 §7)
    _VALID_JSON_ESCAPE: ClassVar[frozenset[str]] = frozenset('"\\\\/bfnrtu')

    # 合法但会被误认为 LaTeX 命令前缀的 JSON 转义 (如 \not 中的 \n, \text 中的 \t)
    _AMBIGUOUS_JSON_ESCAPE: ClassVar[frozenset[str]] = frozenset("ntfrb")
    _KNOWN_LATEX_COMMANDS: ClassVar[frozenset[str]] = frozenset(
        {
            "not",
            "nabla",
            "nu",
            "newcommand",
            "text",
            "theta",
            "times",
            "forall",
            "beta",
            "neq",
            "neg",
            "bar",
            "begin",
            "boxed",
            "binom",
            "frac",
            "big",
            "bigg",
            "rho",
            "mathrm",
            "mathbf",
        }
    )

    @staticmethod
    def _fix_latex_escapes_in_json(json_str: str) -> str:
        """
        修复 JSON 字符串内 LaTeX 命令反斜杠导致的解析失败.

        问题:
        1. \\m, \\e, \\R 等不是合法 JSON 转义 → json.loads 直接崩溃
        2. \\n, \\t, \\f, \\r, \\b 是合法转义, 但在 \\not \\text \\forall
           \\beta \\neq \\neg 中是 LaTeX 命令前缀, 会被错误解析为控制字符

        策略: 扫描 JSON, 在字符串内部:
        - \\x 且 x 不在合法转义集中 → 必然修复
        - \\x 且 x 在 {n,t,f,r,b} 且后续明确组成常见 LaTeX 命令 → 修复
        """
        result: list[str] = []
        in_string = False
        escape = False
        i = 0
        n = len(json_str)

        while i < n:
            ch = json_str[i]
            result.append(ch)

            if in_string:
                if escape:
                    if ch not in BaseAgent._VALID_JSON_ESCAPE:
                        # 非法转义: 必定修复
                        result.insert(-1, "\\")
                    elif (
                        ch in BaseAgent._AMBIGUOUS_JSON_ESCAPE
                        and i + 1 < n
                        and any(
                            (ch + re.match(r"[A-Za-z]*", json_str[i + 1 :]).group(0))
                            .lower()
                            .startswith(command)
                            for command in BaseAgent._KNOWN_LATEX_COMMANDS
                        )
                    ):
                        # 只有明确匹配常见 LaTeX 命令才修复；普通 JSON 的
                        # ``\n中文``/``\nnext`` 必须保留为换行，不能仅因
                        # 后面紧跟字母就被误改成字面反斜杠。
                        result.insert(-1, "\\")
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True

            i += 1

        return "".join(result)

    @staticmethod
    def _escape_control_chars_in_json(json_str: str) -> str:
        """
        修复 JSON 字符串内部的裸控制字符 (未转义的换行/制表符/回车等).

        LLM 在长中文 JSON 输出里常把 prompt 等长字符串直接写成多行,
        而合法 JSON 字符串内不允许出现 ord<0x20 的裸控制字符,
        json.loads 会报 "Invalid control character"。逐个扫描, 在字符串内部
        把裸控制字符替换为 \\uXXXX 转义; 引号/反斜杠转义结构保持不变。
        """
        result: list[str] = []
        in_string = False
        escape = False
        for ch in json_str:
            if in_string:
                if escape:
                    result.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    result.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    result.append(ch)
                    in_string = False
                    continue
                if ord(ch) < 0x20:
                    result.append(f"\\u{ord(ch):04x}")
                    continue
                result.append(ch)
                continue
            # 字符串外
            if ch == '"':
                in_string = True
            result.append(ch)
        return "".join(result)

    @staticmethod
    def _escape_unescaped_quotes_in_json(json_str: str) -> str:
        """修复 LLM 在 JSON 字符串中直接写入的引号。

        澄清结果中的 Markdown 经常包含 ``"展开"`` 这样的说明。模型虽然
        正确返回了对象结构，却忘记转义字符串内部的引号；根据引号后的
        下一个非空字符判断对象/数组分隔符，可以在不改动合法 ``\\"`` 的
        前提下恢复这类常见响应。
        """

        result: list[str] = []
        in_string = False
        escape = False
        length = len(json_str)
        for index, ch in enumerate(json_str):
            if not in_string:
                result.append(ch)
                if ch == '"':
                    in_string = True
                continue
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == "\\":
                result.append(ch)
                escape = True
                continue
            if ch != '"':
                result.append(ch)
                continue
            next_index = index + 1
            while next_index < length and json_str[next_index].isspace():
                next_index += 1
            next_char = json_str[next_index] if next_index < length else ""
            closes_string = next_char in {"", ":", "}", "]"}
            if next_char == ",":
                after_comma = json_str[next_index + 1 :].lstrip()
                # 逗号后既可能是对象的下一个 key，也可能是数组的下一个
                # 字符串/对象/数字。不能只识别 ``"key":``，否则当数组
                # 中某个 LaTeX 字符串需要修复时，会把合法的 ``"a","b"``
                # 误判成一个未闭合字符串。
                closes_string = bool(
                    after_comma.startswith(("}", "]", "{", "["))
                    or re.match(r"(?:true|false|null)(?:\s*[,}\]])", after_comma)
                    or re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", after_comma)
                    or re.match(r'"(?:\\.|[^"\\])*"\s*(?::|[,}\]])', after_comma)
                )
            if closes_string:
                result.append(ch)
                in_string = False
            else:
                result.extend(("\\", ch))
        return "".join(result)

    @staticmethod
    def _close_truncated_json(json_str: str) -> str:
        """补齐仅缺失末尾容器闭合符的 JSON.

        部分兼容端点会在模型输出已经完成字符串值后, 丢掉最后的 ``}``
        或 ``]``。这里只补齐括号配平, 不尝试猜测缺失的字符串内容; 如果
        字符串本身没有闭合, 原文保持不变, 交给上层重试。
        """
        stack: list[str] = []
        in_string = False
        escape = False

        for ch in json_str:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if not stack or stack[-1] != ch:
                    return json_str
                stack.pop()

        if in_string or not stack:
            return json_str
        return json_str + "".join(reversed(stack))

    @staticmethod
    def _extract_json(
        text: str,
        *,
        expected_type: Literal["any", "object", "array"] = "any",
    ) -> str:
        """
        从模型输出中提取 JSON.

        支持三种情况:
        1. ```json ... ``` / ``` ... ``` 代码块包裹
        2. 散文 + JSON 混合 (如 "Sure! Here is the JSON:\\n{...}")
        3. 裸 JSON

        使用括号配平扫描, 正确处理字符串内的括号. ``expected_type`` 用于
        消除散文中的数学方括号/花括号造成的歧义; 例如 ``[-5, 5]`` 可能
        出现在 JSON 对象之前, 但调用方实际需要的是后面的 ``{...}``.
        """
        text = text.strip()

        # 先尝试去 markdown 代码块
        if "```" in text:
            fence_start = text.find("```")
            # 跳过语言标识行
            nl_after = text.find("\n", fence_start)
            if nl_after != -1:
                fence_end = text.find("```", nl_after + 1)
                if fence_end != -1:
                    inner = text[nl_after + 1 : fence_end].strip()
                    if inner:
                        return inner

        # 在原文中找第一个平衡的 JSON 容器. 结构化对象调用方必须优先找
        # ``{...}``, 否则数学说明中的 ``[-5, 5]`` 会被误当成响应主体.
        opening = None if expected_type == "any" else ("{" if expected_type == "object" else "[")
        return BaseAgent._find_balanced_json(text, opening=opening)

    @staticmethod
    def _find_balanced_json(text: str, *, opening: str | None = None) -> str:
        """扫描出第一个平衡的 JSON 对象/数组子串; 找不到则返回原文。

        数学 Markdown 经常在真正的 JSON 前出现 ``x^{1/2}``、``\\boxed{...}``
        等花括号。不能把这些文本片段当作 JSON 起点；对象候选必须以 JSON
        的字符串键（或空对象）开始。若候选确实像 JSON 但被截断，仍返回
        它的剩余文本，交给上层的控制字符/括号修复逻辑处理。
        """
        openings = opening or "{["
        fallback_start: int | None = None

        for start, open_ch in enumerate(text):
            if open_ch not in openings:
                continue

            if open_ch == "{":
                remainder = text[start + 1 :].lstrip()
                if remainder and remainder[0] not in {'"', "}"}:
                    # 跳过 LaTeX/自然语言中的 {内容}，继续寻找真正的对象。
                    continue

            fallback_start = start if fallback_start is None else fallback_start
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            in_string = False
            escape = False

            for index in range(start, len(text)):
                ch = text[index]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            return text[start : index + 1]

        # 未平衡 (模型截断), 返回第一个看起来像 JSON 的候选到结尾。
        return text[fallback_start:] if fallback_start is not None else text

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """
        从模型输出中提取 Python 代码块.

        容错:
        - 支持 ```python ... ``` / ``` ... ``` 包裹
        - 闭合 fence 缺失时 (模型在 max_tokens 处截断) 返回剩余内容而非崩溃
        - 无 fence 时返回原文
        """
        text = text.strip()

        # 优先匹配 ```python
        for marker in ("```python", "```py", "```"):
            idx = text.find(marker)
            if idx == -1:
                continue
            # 跳过 marker 和可能的语言标识行
            nl = text.find("\n", idx)
            if nl == -1:
                continue
            rest = text[nl + 1 :]
            close = rest.find("```")
            if close == -1:
                # 闭合 fence 缺失: 返回剩余内容 (截断代码仍可用)
                return rest.strip()
            return rest[:close].strip()

        return text
