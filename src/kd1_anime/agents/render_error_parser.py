"""从 Manim/Slurm 日志提取可执行的渲染错误证据。

渲染日志经常包含多个重试、Rich traceback 和很长的环境输出。把最后一段
traceback 归一化为小型证据对象，可以让 AutoFix 只关注真正的失败位置，也
可以用稳定指纹判断修复是否取得了进展。本模块只解析文本，不执行日志中的
任何 Python 代码。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Literal

from kd1_anime.security import redact_text

ErrorCategory = Literal[
    "latex",
    "renderer",
    "python",
    "lifecycle",
    "timeout",
    "infrastructure",
    "unknown",
]

_TRACEBACK_MARKER = "Traceback (most recent call last):"
_FILE_RE = re.compile(r'^\s*File\s+["\'](?P<file>.*?)["\'],\s+line\s+(?P<line>\d+)')
_EXCEPTION_RE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))(?::\s*(?P<message>.*))?\s*$"
)
_RUN_ID_RE = re.compile(r"\b\d{8}-\d{6}-[0-9a-f]{8}\b", re.IGNORECASE)
_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")


@dataclass(frozen=True, slots=True)
class RenderErrorEvidence:
    """一次渲染错误的脱敏、可持久化证据。"""

    error_type: str
    message: str
    category: ErrorCategory
    file: str = ""
    line: int | None = None
    code_line: str = ""
    source_context: str = ""
    traceback: str = ""
    fingerprint: str = ""

    def to_dict(self) -> dict[str, object]:
        """返回适合写入 JSON artifact 的普通字典。"""

        return asdict(self)

    def prompt_text(self) -> str:
        """构造短证据块，避免把整份日志重复注入修复提示词。"""

        location = self.file or "<unknown>"
        if self.line is not None:
            location += f":{self.line}"
        lines = [
            f"异常类型: {self.error_type or '未知'}",
            f"分类: {self.category}",
            f"位置: {location}",
            f"报错消息: {self.message or '未知'}",
        ]
        if self.code_line:
            lines.append(f"traceback 代码行: {self.code_line}")
        if self.source_context:
            lines.append(f"源码上下文:\n{self.source_context}")
        if self.fingerprint:
            lines.append(f"稳定错误指纹: {self.fingerprint}")
        return "\n".join(lines)


def _normalise_file(path: str) -> str:
    value = path.strip()
    if not value or value.startswith("<"):
        return value
    # 日志可能带有提交机上的绝对路径。AutoFix 只需要场景文件名，避免把
    # 用户名、工作目录和临时目录写入跨 Agent 上下文。
    return PurePath(value).name


def _traceback_block(log: str) -> str:
    blocks = log.split(_TRACEBACK_MARKER)
    if len(blocks) == 1:
        return log
    return _TRACEBACK_MARKER + blocks[-1]


def _last_exception(lines: list[str]) -> tuple[str, str]:
    found: tuple[str, str] = ("", "")
    for line in lines:
        match = _EXCEPTION_RE.match(line)
        if match:
            found = (match.group("type"), (match.group("message") or "").strip())
    return found


def _category(error_type: str, message: str, *, renderer: str | None) -> ErrorCategory:
    text = f"{error_type} {message}".lower()
    if any(token in text for token in ("latex", "xelatex", "missing $", "emergency stop", "tex/")):
        return "latex"
    if any(token in text for token in ("timeout", "time limit", "timed out", "killed")):
        return "timeout"
    if any(
        token in text
        for token in (
            "conda",
            "slurm",
            "permission denied",
            "out of memory",
            "oom",
            "node_fail",
            "no such file or directory",
            "command not found",
        )
    ):
        return "infrastructure"
    if any(
        token in text
        for token in ("should_render", "openglcamera", "camera.frame", "egl", "glx", "renderer")
    ) or (renderer == "opengl" and "camera" in text):
        return "renderer"
    if any(token in text for token in ("active", "transform", "fadeout", "vgroup", "lifecycle")):
        return "lifecycle"
    if error_type:
        return "python"
    return "unknown"


def _stable_fingerprint(error_type: str, message: str, code_line: str, category: str) -> str:
    value = f"{category}|{error_type}|{message}|{code_line}".lower()
    value = _RUN_ID_RE.sub("<run>", value)
    value = _HEX_RE.sub("<hex>", value)
    value = _NUMBER_RE.sub("#", value)
    value = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def extract_render_error(
    error_log: str,
    *,
    source_code: str = "",
    renderer: str | None = None,
    secrets: tuple[str, ...] = (),
    max_traceback_chars: int = 12_000,
) -> RenderErrorEvidence:
    """提取最后一个 traceback，并尽量绑定到源码行。

    ``error_log`` 可能不是标准 traceback（例如只有 Slurm 的 TIMEOUT 行），
    因此所有字段都允许为空，调用方可以依据 ``category`` 和指纹选择回退
    路径。日志和源码上下文均会脱敏且有长度上限。
    """

    safe_log = redact_text(error_log or "", secrets)
    block = _traceback_block(safe_log)
    block = block[-max_traceback_chars:]
    lines = block.splitlines()
    file_name = ""
    line_number: int | None = None
    code_line = ""
    for index, line in enumerate(lines):
        match = _FILE_RE.match(line)
        if not match:
            continue
        file_name = _normalise_file(match.group("file"))
        line_number = int(match.group("line"))
        # Python traceback 的下一行是实际源码；Rich traceback 也通常保留它。
        if index + 1 < len(lines):
            candidate = lines[index + 1].strip()
            if candidate and not candidate.startswith("File "):
                code_line = candidate[:2_000]

    error_type, message = _last_exception(lines)
    if not error_type:
        # 没有标准异常行时，保留最后一条有内容的诊断行，不能把整段日志
        # 误当作异常消息，也不能执行它。
        non_empty = [line.strip() for line in lines if line.strip()]
        message = (non_empty[-1] if non_empty else "")[:3_000]
    category = _category(error_type, message, renderer=renderer)

    context = ""
    source_lines = source_code.splitlines()
    if line_number is not None and 1 <= line_number <= len(source_lines):
        start = max(1, line_number - 2)
        end = min(len(source_lines), line_number + 2)
        context = "\n".join(
            f"{number:>4}: {source_lines[number - 1]}" for number in range(start, end + 1)
        )
        context = redact_text(context, secrets)[:4_000]

    traceback = redact_text(block, secrets)
    fingerprint = _stable_fingerprint(error_type, message, code_line, category)
    return RenderErrorEvidence(
        error_type=error_type[:300],
        message=message[:3_000],
        category=category,
        file=file_name[:300],
        line=line_number,
        code_line=code_line,
        source_context=context,
        traceback=traceback,
        fingerprint=fingerprint,
    )


__all__ = ["ErrorCategory", "RenderErrorEvidence", "extract_render_error"]
