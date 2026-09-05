"""本地匿名动画配方的保存与去重。

配方不保存用户原始提示词、运行目录或外部服务凭据，只保存已经通过
确定性校验/渲染的代码片段及少量实现标签。文件使用 Markdown，因而可以
直接作为普通 RAG 知识源参与检索。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.config import resolve_runtime_path, settings
from kd1_anime.security import redact_text

_RUN_ID_RE = re.compile(r"\b\d{8}-\d{6}-[0-9a-f]{8}\b", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/(?:home|root|tmp|var|mnt|opt|data|scratch|workspaces?|projects?)/[^\s'\"`<>]+|[A-Za-z]:\\[^\s'\"`<>]+)"
)
_SENSITIVE_LINE_RE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|authorization|bearer|secret|password|private\s+key)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)


def anonymize_code(code: str) -> str:
    """去除运行身份、绝对路径和疑似凭据行，保留可复用代码结构。"""

    safe_lines: list[str] = []
    for raw_line in str(code or "").splitlines():
        if _SENSITIVE_LINE_RE.search(raw_line):
            continue
        safe_line = _RUN_ID_RE.sub("<run>", raw_line)
        safe_line = _ABSOLUTE_PATH_RE.sub("<path>", safe_line)
        safe_lines.append(_SECRET_VALUE_RE.sub("<redacted>", redact_text(safe_line)).rstrip())
    return "\n".join(safe_lines).strip() + ("\n" if safe_lines else "")


def _safe_label(value: str, *, limit: int = 300) -> str:
    value = _SECRET_VALUE_RE.sub(
        "<redacted>", redact_text(_RUN_ID_RE.sub("<run>", str(value or "")))
    )
    value = _ABSOLUTE_PATH_RE.sub("<path>", value)
    return " ".join(value.split())[:limit]


class RecipeRecord(BaseModel):
    """配方元数据；不包含用户提示词和服务凭据。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    recipe_id: str = Field(pattern=r"^recipe-[0-9a-f]{16,64}$")
    created_at: datetime
    renderer: str = Field(min_length=1, max_length=20)
    semantic_intent: str = Field(default="", max_length=500)
    object_kinds: list[str] = Field(default_factory=list, max_length=40)
    capabilities: list[str] = Field(default_factory=list, max_length=40)
    verification: str = Field(default="rendered", max_length=30)
    code: str = Field(min_length=1, max_length=80_000)
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecipeStore:
    """将匿名配方以私有 Markdown 文件写入本地知识库。"""

    def __init__(self, root: Path | None = None) -> None:
        configured = root if root is not None else settings.RAG_RECIPES_DIR
        self.root = resolve_runtime_path(configured) if configured is not None else None

    def _ensure_root(self) -> Path:
        if self.root is None:
            raise ValueError("未配置 RAG_RECIPES_DIR")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        return self.root

    @staticmethod
    def _recipe_id(code: str) -> str:
        return "recipe-" + hashlib.sha256(code.encode("utf-8")).hexdigest()

    def save(
        self,
        code: str,
        *,
        renderer: str,
        semantic_intent: str = "",
        object_kinds: list[str] | tuple[str, ...] = (),
        capabilities: list[str] | tuple[str, ...] = (),
        verification: str = "rendered",
    ) -> tuple[RecipeRecord, Path, bool]:
        """保存配方并按代码哈希去重，返回 ``(record, path, created)``。"""

        safe_code = anonymize_code(code)
        if not safe_code:
            raise ValueError("代码脱敏后为空，不能保存配方")
        recipe_id = self._recipe_id(safe_code)
        root = self._ensure_root()
        safe_renderer = re.sub(r"[^A-Za-z0-9_.-]", "_", _safe_label(renderer, limit=20))[:20]
        safe_renderer = safe_renderer or "unknown"
        path = root / f"{recipe_id}-{safe_renderer}.md"
        if path.is_symlink():
            raise ValueError(f"Recipe 目标不能是符号链接: {path}")
        record = RecipeRecord(
            recipe_id=recipe_id,
            created_at=datetime.now(timezone.utc),
            renderer=safe_renderer,
            semantic_intent=_safe_label(semantic_intent),
            object_kinds=sorted({_safe_label(item, limit=80) for item in object_kinds if item}),
            capabilities=sorted({_safe_label(item, limit=80) for item in capabilities if item}),
            verification=_safe_label(verification, limit=30) or "rendered",
            code=safe_code,
            code_sha256=hashlib.sha256(safe_code.encode("utf-8")).hexdigest(),
        )
        if path.is_file():
            return record, path, False
        metadata = [
            "# Anonymous animation recipe",
            "",
            f"- renderer: {record.renderer}",
            f"- verification: {record.verification}",
            f"- semantic_intent: {record.semantic_intent or 'general scene'}",
            f"- object_kinds: {', '.join(record.object_kinds) or 'unspecified'}",
            f"- capabilities: {', '.join(record.capabilities) or 'unspecified'}",
            f"- code_sha256: {record.code_sha256}",
            "",
            "```python",
            record.code.rstrip("\n"),
            "```",
            "",
        ]
        payload = "\n".join(metadata)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".recipe-", suffix=".tmp", dir=root)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # 同一代码哈希对应同一文件名；replace 对竞争写入是原子的，
            # 两个写入者的内容必然相同。
            os.replace(temporary, path)
            path.chmod(0o600)
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return record, path, True
        finally:
            if descriptor != -1:
                with suppress(OSError):
                    os.close(descriptor)
            temporary.unlink(missing_ok=True)


__all__ = ["RecipeRecord", "RecipeStore", "anonymize_code"]
