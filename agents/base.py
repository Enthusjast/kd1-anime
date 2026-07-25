"""
BaseAgent 基类
封装 OpenAI 兼容 API 调用,提供:
- 指数退避重试机制
- 结构化 JSON 输出 (支持 Pydantic 模型校验)
- Rich 控制台美化输出 (Agent 思考过程)
- 健壮的 JSON / 代码块提取 (容错散文包裹、fence、截断)
"""

import json
import time
from typing import Type, TypeVar

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    BadRequestError,
    RateLimitError,
    APITimeoutError,
)
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel

from config import settings

T = TypeVar("T", bound=BaseModel)

console = Console()


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
                timeout=httpx.Timeout(120.0, connect=30.0, read=120.0),
            )
        return self._client

    def _log(self, message: str, style: str = "bold cyan") -> None:
        """打印 Agent 思考过程"""
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
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        json_fallback_used = False
        temp_fallback_used = False
        max_tokens_fallback_used = False

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
                if stream:
                    content = self._stream_llm(kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    content = content.strip()
                if not content:
                    finish = ""
                    if not stream:
                        finish = getattr(response.choices[0], "finish_reason", "N/A")
                    else:
                        finish = "stream_empty"
                    if finish == "length":
                        raise RuntimeError(
                            f"[{self.name}] LLM 输出被 max_tokens={kwargs.get('max_tokens', '?' )} "
                            "截断且内容为空. 请在 .env 中增大 LLM_MAX_TOKENS (建议 8192+)."
                        )
                    self._log(
                        f"LLM 返回空响应, 将重试... (finish_reason={finish})",
                        style="bold yellow",
                    )
                    last_error = RuntimeError("LLM 返回空响应")
                    delay = settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)
                    continue
                # 有内容: 即使是 finish_reason=length (截断) 也直接返回, 不重试
                if getattr(settings, "LLM_DEBUG", False) and not stream:
                    preview = content[:500] + ("..." if len(content) > 500 else "")
                    console.print(f"[dim]DEBUG [response]: {preview}[/]", markup=False)
                return content

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                # 关闭 client 重建, 避免 httpx 连接池复用死连接
                self._client = None
                delay = settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._log(f"API 限流/超时/连接错误, {delay:.1f}s 后重试... ({e})", style="bold yellow")
                time.sleep(delay)

            except BadRequestError as e:
                # 400 通常是确定性错误,不应重试. 但若由 response_format 引起,
                # 降级为"只用 prompt 要求 JSON"再试一次 (不消耗额外重试预算外的一次).
                msg = str(e).lower()
                if (
                    json_mode
                    and not json_fallback_used
                    and ("response_format" in msg or "json" in msg or "format" in msg)
                ):
                    self._log("端点不支持 response_format, 降级为 prompt-only JSON 重试", style="yellow")
                    kwargs.pop("response_format", None)
                    json_fallback_used = True
                    attempt -= 1  # 参数修复, 不消耗重试次数
                    continue
                # Kimi 等推理模型只允许 temperature=1
                if (
                    not temp_fallback_used
                    and ("temperature" in msg or "only 1 is allowed" in msg)
                ):
                    self._log("模型仅支持 temperature=1, 降级重试", style="yellow")
                    kwargs["temperature"] = 1.0
                    temp_fallback_used = True
                    attempt -= 1  # 参数修复, 不消耗重试次数
                    continue
                # 模型拒绝 max_tokens 参数
                if (
                    not max_tokens_fallback_used
                    and "max_tokens" in msg
                    and "max_tokens" in kwargs
                ):
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
                    f"API 错误 (status={e.status_code}, type={e.type}): {e.message}",
                    style="bold yellow",
                )
                # 请求被拒绝 / 被封 — 不可重试
                blocked_keywords = ["blocked", "rejected", "denied", "forbidden"]
                msg_lower = str(e).lower()
                if any(kw in msg_lower for kw in blocked_keywords):
                    raise RuntimeError(
                        f"[{self.name}] API 请求被拒绝 (不可重试): {e}"
                    ) from e
                last_error = e
                delay = settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"[{self.name}] LLM 调用发生未知错误: {e}") from e

        raise RuntimeError(
            f"[{self.name}] LLM 调用在 {settings.LLM_MAX_RETRIES} 次重试后仍然失败: {last_error}"
        )

    def _stream_llm(self, kwargs: dict) -> str:
        """流式调用 LLM, 边生成边打印, 收集完整文本返回.
        按 ESC 可提前截断 (保留已生成内容).
        空响应返回 "" 而非抛异常, 由 call_llm 统一处理重试."""
        import sys
        import threading
        import select

        kwargs = {**kwargs, "stream": True}
        chunks: list[str] = []
        reasoning_chunks = 0
        content_chunks = 0
        empty_chunks = 0

        # --- ESC 检测线程 ---
        cancelled = threading.Event()
        esc_buffer: list[bytes] = []

        def _watch_esc() -> None:
            """监控 stdin, 检测裸 ESC (非 Esc+Enter 等组合键)"""
            try:
                while not cancelled.is_set():
                    r, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if not r:
                        continue
                    b = sys.stdin.buffer.read(1)
                    if not b:
                        break
                    if b == b"\x1b":
                        esc_buffer.append(b)
                        # 等 80ms 看是否有后续字节 (组合键如 Esc+Enter)
                        import time as _time
                        _time.sleep(0.08)
                        r2, _, _ = select.select([sys.stdin], [], [], 0)
                        if r2:
                            # 有后续字节 → 是组合键，不触发取消
                            esc_buffer.clear()
                            sys.stdin.buffer.read(1)  # 吞掉后续字节
                        else:
                            # 裸 ESC → 取消
                            cancelled.set()
                            break
            except (OSError, ValueError):
                pass

        watcher = threading.Thread(target=_watch_esc, daemon=True)
        watcher.start()
        # ---

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if cancelled.is_set():
                console.print("\n[dim](ESC 截断)[/]")
                try:
                    stream.close()
                except Exception:
                    pass
                break

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)
            finish = getattr(chunk.choices[0], "finish_reason", None)
            if reasoning:
                if reasoning_chunks == 0:
                    console.print("[dim]思考:[/] ", end="")
                reasoning_chunks += 1
                if reasoning_chunks % 20 == 0:
                    console.print("[dim].[/]", end="")
            if text:
                content_chunks += 1
                chunks.append(text)
                console.print(text, end="", soft_wrap=True, highlight=False)
            elif not reasoning and not finish:
                empty_chunks += 1
                if empty_chunks <= 3 and getattr(settings, "LLM_DEBUG", False):
                    delta_fields = {
                        k: v for k, v in delta.__dict__.items() if v is not None
                    }
                    console.print(
                        f"\n[dim]DEBUG 空 chunk delta: {delta_fields}[/]", markup=False
                    )
            elif finish:
                from datetime import datetime
                now = datetime.now().strftime("%H:%M:%S")
                console.print(
                    f"\n[dim]({now} 流结束: {finish}, "
                    f"思考 {reasoning_chunks}, 内容 {content_chunks}, 空 {empty_chunks})[/]"
                )

        cancelled.set()  # 通知 watcher 退出
        if content_chunks == 0 and getattr(settings, "LLM_DEBUG", False):
            console.print(
                f"[bold red]诊断:[/] 无 content. reasoning={reasoning_chunks}, 可能是推理模型耗尽 token 或 API 异常."
            )
        if reasoning_chunks > 0 and content_chunks == 0:
            console.print(
                f"[bold yellow]警告:[/] 推理消耗了所有 token, 未生成内容. "
                f"请大幅增大 LLM_MAX_TOKENS (建议 16384+)."
            )
        console.print()
        return "".join(chunks).strip()

    def call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        response_model: Type[T],
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

        # 修复 LLM 输出中 LaTeX 命令导致的非法 JSON 转义 (如 \mathbb, \exists)
        json_str = self._fix_latex_escapes_in_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self._log_panel("JSON 解析失败", f"原始响应:\n{raw}\n\n修复后:\n{json_str}\n\n错误: {e}", style="red")
            raise RuntimeError(f"[{self.name}] LLM 返回了无效的 JSON: {e}") from e

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
        item_model: Type[T],
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

        # 修复 LLM 输出中 LaTeX 命令导致的非法 JSON 转义
        json_str = self._fix_latex_escapes_in_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"[{self.name}] LLM 返回了无效的 JSON: {e}") from e

        # 支持 {"items": [...]} 或直接 [...]
        if isinstance(data, dict) and "items" in data:
            items_data = data["items"]
        elif isinstance(data, list):
            items_data = data
        else:
            raise RuntimeError(
                f"[{self.name}] 期望 JSON 列表或 {{'items': [...]}}, 收到: {type(data)}"
            )

        results = []
        for i, item in enumerate(items_data):
            try:
                results.append(item_model.model_validate(item))
            except ValidationError as e:
                self._log(f"第 {i} 项校验失败,跳过: {e}", style="yellow")
                continue

        if not results:
            raise RuntimeError(f"[{self.name}] 没有任何有效的列表项通过校验")

        return results

    # ------------------------------------------------------------------
    # 提取工具 (健壮版)
    # ------------------------------------------------------------------

    # JSON 合法转义字符 (RFC 8259 §7)
    _VALID_JSON_ESCAPE = set('"\\/bfnrtu')

    # 合法但会被误认为 LaTeX 命令前缀的 JSON 转义 (如 \not 中的 \n, \text 中的 \t)
    _AMBIGUOUS_JSON_ESCAPE = set("ntfrb")

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
                    elif ch in BaseAgent._AMBIGUOUS_JSON_ESCAPE and i + 1 < n:
                        # 合法转义但后跟字母 → LaTeX 命令 (如 \not, \text, \forall)
                        if json_str[i + 1].isalpha():
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
                    inner = text[nl_after + 1:fence_end].strip()
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
                        return text[start:i + 1]
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
            rest = text[nl + 1:]
            close = rest.find("```")
            if close == -1:
                # 闭合 fence 缺失: 返回剩余内容 (截断代码仍可用)
                return rest.strip()
            return rest[:close].strip()

        return text
