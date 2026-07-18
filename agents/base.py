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
            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
            )
        return self._client

    def _log(self, message: str, style: str = "bold cyan") -> None:
        """打印 Agent 思考过程"""
        console.print(f"[{style}][{self.name}][/] {message}")

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

        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            try:
                self._log(f"LLM 调用中... (尝试 {attempt}/{settings.LLM_MAX_RETRIES})")
                if stream:
                    return self._stream_llm(kwargs)
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("LLM 返回空响应")
                return content.strip()

            except (RateLimitError, APITimeoutError) as e:
                last_error = e
                delay = settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._log(f"API 限流/超时, {delay:.1f}s 后重试... ({e})", style="bold yellow")
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
                    continue
                # 其他 400 直接失败, 不重试
                raise RuntimeError(f"[{self.name}] 请求被拒绝 (400): {e}") from e

            except APIError as e:
                last_error = e
                delay = settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._log(f"API 错误, {delay:.1f}s 后重试... ({e})", style="bold yellow")
                time.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"[{self.name}] LLM 调用发生未知错误: {e}") from e

        raise RuntimeError(
            f"[{self.name}] LLM 调用在 {settings.LLM_MAX_RETRIES} 次重试后仍然失败: {last_error}"
        )

    def _stream_llm(self, kwargs: dict) -> str:
        """流式调用 LLM, 边生成边打印, 收集完整文本返回"""
        kwargs = {**kwargs, "stream": True}
        chunks: list[str] = []
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                chunks.append(text)
                console.print(text, end="", soft_wrap=True, highlight=False)
        console.print()  # 换行
        content = "".join(chunks).strip()
        if not content:
            raise RuntimeError("LLM 流式返回空响应")
        return content

    def call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        response_model: Type[T],
        temperature: float | None = None,
    ) -> T:
        """
        调用 LLM 并将响应解析为 Pydantic 模型

        结构化提取默认使用 temperature=0.0 以保证确定性.

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            response_model: 期望的 Pydantic 模型类型
            temperature: 温度参数 (默认 0.0)

        Returns:
            校验后的 Pydantic 模型实例
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
        except json.JSONDecodeError as e:
            self._log_panel("JSON 解析失败", f"原始响应:\n{raw}\n\n错误: {e}", style="red")
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
