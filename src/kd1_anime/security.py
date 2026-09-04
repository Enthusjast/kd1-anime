"""用于诊断输出的轻量敏感信息脱敏工具。

生成流水线会把外部服务异常、用户提示词和模型返回值写入日志/事件文件。
这些内容本身不应被当成可信输入，也不能因为异常路径把 API Key、Bearer
凭据或 URL 查询参数带到持久化文件和终端中。本模块不参与鉴权，只负责在
日志边界做保守脱敏。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

_QUERY_SECRET_PATTERN = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|key)=)"
    r"[^&#\s]+",
    flags=re.IGNORECASE,
)
_HEADER_SECRET_PATTERN = re.compile(
    r"((?:authorization|x-api-key|api-key|token|secret|password)\s*[:=]\s*)"
    r"(?:bearer\s+)?[^\s,;]+",
    flags=re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"(\bbearer\s+)[^\s,;]+",
    flags=re.IGNORECASE,
)
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "x_api_key",
        "authorization",
        "access_token",
        "token",
        "secret",
        "password",
    }
)


def redact_text(value: object, secrets: Iterable[str] = ()) -> str:
    """返回不包含常见凭据的文本。

    ``secrets`` 中的完整值会优先替换；随后再处理 URL 查询参数和常见
    Authorization 表达式。短值不做全文替换，避免把普通单词误删。
    """

    text = str(value).strip()
    unique_secrets = sorted(
        {secret for secret in secrets if isinstance(secret, str) and len(secret) >= 4},
        key=len,
        reverse=True,
    )
    for secret in unique_secrets:
        text = text.replace(secret, "<redacted>")
    text = _QUERY_SECRET_PATTERN.sub(r"\1<redacted>", text)
    text = _HEADER_SECRET_PATTERN.sub(r"\1<redacted>", text)
    text = _BEARER_PATTERN.sub(r"\1<redacted>", text)
    return text


def redact_value(value: Any, secrets: Iterable[str] = ()) -> Any:
    """递归脱敏字符串，同时尽量保留回调数据的原始容器类型。"""

    secrets = tuple(secrets)
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            result[key] = (
                "<redacted>"
                if normalized_key in _SENSITIVE_KEY_NAMES
                else redact_value(item, secrets)
            )
        return result
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets) for item in value)
    return value


def to_jsonable(value: Any) -> Any:
    """把事件数据转换成 JSON 兼容值，不调用不可信对象的复杂序列化逻辑。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(mode="json"))
        except (TypeError, ValueError):
            return redact_text(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return redact_text(value)


def redact_jsonable(value: Any, secrets: Iterable[str] = ()) -> Any:
    """先转换为 JSON 兼容结构，再递归脱敏。"""

    return redact_value(to_jsonable(value), secrets)
