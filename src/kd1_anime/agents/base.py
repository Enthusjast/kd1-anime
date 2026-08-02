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
import time
from contextlib import suppress
from typing import ClassVar, TypeVar

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

from kd1_anime.config import settings

T = TypeVar("T", bound=BaseModel)

console = Console()


class StreamCancelledError(RuntimeError):
    """用户主动取消流式响应。"""


class BaseAgent:
    """所有 Agent 的基类,封装 LLM 交互逻辑"""

    name: str = "BaseAgent"

    def __init__(self):
        # 延迟构造 client: 避免 API Key 为空时在实例化阶段就崩溃
        # (让调用方有机会先给出友好的配置错误提示)
        self._client: OpenAI | None = None
        self.model = settings.LLM_MODEL

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI client"""
        if self._client is None:
            import httpx

            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                timeout=httpx.Timeout(
                    settings.LLM_TIMEOUT_READ,
                    connect=settings.LLM_TIMEOUT_CONNECT,
                    read=settings.LLM_TIMEOUT_READ,
                ),
            )
        return self._client

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
        """用 Panel 展示详细内容"""
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

        Returns:
            LLM 的文本响应

        Raises:
            RuntimeError: 重试耗尽后抛出
        """
        settings.require_llm_key()

        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        # 部分 (推理类) 模型拒绝 max_tokens; 通过设置可关闭
        if getattr(settings, "LLM_SEND_MAX_TOKENS", True) and tokens:
            kwargs["max_tokens"] = tokens
        if json_mode and getattr(settings, "LLM_USE_JSON_MODE", True):
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        json_fallback_used = False
        stream_fallback_used = False
        # 静默流式: stream=False(不展示内容)时仍使用流式传输, 避免长生成超时
        use_stream_transport = stream or (
            not stream and getattr(settings, "LLM_SILENT_STREAM", False)
        )
        temp_fallback_used = False
        max_tokens_fallback_used = False
        max_tokens_boosted = False

        attempt = 0
        while attempt < settings.LLM_MAX_RETRIES:
            attempt += 1
            try:
                self._log(f"LLM 调用中... (尝试 {attempt}/{settings.LLM_MAX_RETRIES})")
                if getattr(settings, "LLM_DEBUG", False):
                    for i, msg in enumerate(messages):
                        role = msg["role"]
                        body = msg["content"]
                        preview = body[:500] + ("..." if len(body) > 500 else "")
                        console.print(f"[dim]DEBUG [{role} #{i}]: {preview}[/]", markup=False)
                if use_stream_transport:
                    content, finish_reason = self._stream_llm(kwargs, display=stream)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    content = content.strip()
                if not content:
                    finish = finish_reason or "stream_empty"
                    if finish == "length":
                        max_tokens_val = kwargs.get('max_tokens')
                        if max_tokens_val:
                            raise RuntimeError(
                                f"[{self.name}] LLM 输出被 max_tokens={max_tokens_val} "
                                "截断且内容为空. 请在 .env 中增大 LLM_MAX_TOKENS (建议 8192+)."
                            )
                        else:
                            raise RuntimeError(
                                f"[{self.name}] LLM 输出被截断且内容为空. "
                                "请尝试简化输入或检查模型能力."
                            )
                    if json_mode and "response_format" in kwargs and not json_fallback_used:
                        self._log(
                            "端点在 response_format 模式返回空内容，"
                            "降级为 prompt-only JSON 重试",
                            style="yellow",
                        )
                        kwargs.pop("response_format", None)
                        json_fallback_used = True
                        # 在系统提示中添加明确的 JSON 输出要求
                        for msg in kwargs.get("messages", []):
                            if msg.get("role") == "system":
                                msg["content"] += "\n\n重要：你必须返回有效的 JSON 格式。不要包含任何其他文本，只返回 JSON。"
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
                    if "max_tokens" not in kwargs and not max_tokens_boosted:
                        boost = settings.LLM_EMPTY_RETRY_MAX_TOKENS
                        self._log(
                            f"空响应: 补充 max_tokens={boost} 后重试 "
                            "(推理模型可能耗尽输出预算)",
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
                    if attempt < settings.LLM_MAX_RETRIES:
                        delay = self._retry_delay(attempt, last_error)
                        time.sleep(delay)
                    continue
                # 有内容: 即使是 finish_reason=length (截断) 也直接返回, 不重试
                if getattr(settings, "LLM_DEBUG", False) and (
                    not use_stream_transport or not stream
                ):
                    preview = content[:500] + ("..." if len(content) > 500 else "")
                    console.print(f"[dim]DEBUG [response]: {preview}[/]", markup=False)
                return content

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                # 关闭 client 重建, 避免 httpx 连接池复用死连接
                self._client = None
                if attempt < settings.LLM_MAX_RETRIES:
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
                            msg_item["content"] += "\n\n重要：你必须返回有效的 JSON 格式。不要包含任何其他文本，只返回 JSON。"
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
                if attempt < settings.LLM_MAX_RETRIES:
                    delay = self._retry_delay(attempt, e)
                    time.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"[{self.name}] LLM 调用发生未知错误: {e}") from e

        raise RuntimeError(
            f"[{self.name}] LLM 调用在 {settings.LLM_MAX_RETRIES} 次重试后仍然失败: {last_error}"
        )

    @staticmethod
    def _retry_delay(attempt: int, error: Exception | None = None) -> float:
        """指数退避加随机抖动，并优先尊重服务端 Retry-After。"""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(300.0, max(0.1, float(retry_after)))
            except (TypeError, ValueError):
                pass
        base = settings.LLM_RETRY_BASE_DELAY * (2 ** max(0, attempt - 1))
        return min(300.0, base + random.uniform(0, settings.LLM_RETRY_BASE_DELAY))

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
                    if display and empty_chunks <= 3 and settings.LLM_DEBUG:
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
        if display and content_chunks == 0 and settings.LLM_DEBUG:
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
    ) -> T:
        """
        调用 LLM 并将响应解析为 Pydantic 模型

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            response_model: 期望的 Pydantic 模型类型
            temperature: 温度参数 (默认 0.0)
            stream: 是否使用流式传输 (部分 API 非流式可能超时)
        """
        temp = 0.0 if temperature is None else temperature

        raw = self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temp,
            json_mode=True,
            stream=stream,
        )

        json_str = self._extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as first_error:
            repaired = self._fix_latex_escapes_in_json(json_str)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as error:
                self._log_panel(
                    "JSON 解析失败",
                    f"原始响应:\n{raw}\n\n修复后:\n{repaired}\n\n错误: {error}",
                    style="red",
                )
                raise RuntimeError(f"[{self.name}] LLM 返回了无效的 JSON: {first_error}") from error

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
            self._log_panel(
                "Pydantic 校验失败",
                f"JSON 数据:\n{json.dumps(data, ensure_ascii=False, indent=2)}\n\n错误: {e}",
                style="red",
            )
            raise RuntimeError(f"[{self.name}] LLM 输出不符合预期结构: {e}") from e

    def call_llm_json_list(
        self,
        system_prompt: str,
        user_message: str,
        item_model: type[T],
        temperature: float | None = None,
    ) -> list[T]:
        """
        调用 LLM 并将响应解析为 Pydantic 模型列表

        期望 LLM 返回 {"items": [...]} 或直接返回 [...]
        """
        temp = 0.0 if temperature is None else temperature

        raw = self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temp,
            json_mode=True,
        )

        json_str = self._extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as first_error:
            repaired = self._fix_latex_escapes_in_json(json_str)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"[{self.name}] LLM 返回了无效的 JSON: {first_error}") from error

        # 支持 {"items": [...]} 或直接 [...]
        if isinstance(data, dict) and "items" in data:
            items_data = data["items"]
        elif isinstance(data, list):
            items_data = data
        else:
            raise RuntimeError(
                f"[{self.name}] 期望 JSON 列表或 {{'items': [...]}}, 收到: {type(data)}"
            )

        if not isinstance(items_data, list):
            raise RuntimeError(f"[{self.name}] items 必须是 JSON 数组")

        results = []
        validation_errors: list[str] = []
        for i, item in enumerate(items_data):
            try:
                results.append(item_model.model_validate(item))
            except ValidationError as e:
                validation_errors.append(f"第 {i} 项: {e}")

        if validation_errors:
            preview = "\n".join(validation_errors[:5])
            raise RuntimeError(
                f"[{self.name}] 列表中有 {len(validation_errors)} 项未通过结构校验，"
                f"拒绝使用残缺结果：\n{preview}"
            )

        if not results:
            raise RuntimeError(f"[{self.name}] 没有任何有效的列表项通过校验")

        return results

    # ------------------------------------------------------------------
    # 提取工具 (健壮版)
    # ------------------------------------------------------------------

    # JSON 合法转义字符 (RFC 8259 §7)
    _VALID_JSON_ESCAPE: ClassVar[frozenset[str]] = frozenset('"\\\\/bfnrtu')

    # 合法但会被误认为 LaTeX 命令前缀的 JSON 转义 (如 \not 中的 \n, \text 中的 \t)
    _AMBIGUOUS_JSON_ESCAPE: ClassVar[frozenset[str]] = frozenset("ntfrb")

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
        - \\x 且 x 在 {n,t,f,r,b} 且后一个字符是字母 → LaTeX, 修复
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
                        and json_str[i + 1].isalpha()
                    ):
                        # 合法转义但后跟字母 → LaTeX 命令 (如 \not, \text, \forall)
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
    def _extract_json(text: str) -> str:
        """
        从模型输出中提取 JSON.

        支持三种情况:
        1. ```json ... ``` / ``` ... ``` 代码块包裹
        2. 散文 + JSON 混合 (如 "Sure! Here is the JSON:\\n{...}")
        3. 裸 JSON

        使用括号配平扫描, 正确处理字符串内的括号.
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

        # 在原文中找第一个平衡的 {...} 或 [...]
        return BaseAgent._find_balanced_json(text)

    @staticmethod
    def _find_balanced_json(text: str) -> str:
        """扫描出第一个平衡的 JSON 对象/数组子串; 找不到则返回原文"""
        start = -1
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                break
        if start == -1:
            return text

        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
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
                        return text[start : i + 1]
        # 未平衡 (模型截断), 返回从 start 到结尾
        return text[start:]

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
