"""本地 LLM 响应缓存。

缓存的目标是减少计划审查、代码审查和恢复运行时的重复请求，而不是替代
重试逻辑。数据库只保存不可逆的请求指纹、完整文本响应和少量诊断元数据；
绝不保存 API Key。文件和目录权限在每次打开时都会重新收紧，适合多线程和
同一用户的多个 CLI 进程并发使用。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kd1_anime.config import APP_HOME, LLMRuntimeProfile, settings


@dataclass(frozen=True, slots=True)
class CacheStats:
    """一次进程生命周期内的缓存统计。"""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0


def make_cache_key(
    profile: LLMRuntimeProfile,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    json_mode: bool,
    allow_truncated: bool,
    extra: str = "",
) -> str:
    """生成稳定请求指纹。

    API Key 故意不参与 payload。端点、模型、系统/用户消息、生成参数和
    结构化输出模式都参与计算，避免不同阶段意外复用错误响应。
    """

    payload = {
        "profile": {
            "label": profile.label,
            "env_prefix": profile.env_prefix,
            "base_url": profile.base_url,
            "model": profile.model,
            "send_max_tokens": profile.send_max_tokens,
            "silent_stream": profile.silent_stream,
            "use_json_mode": profile.use_json_mode,
            "trust_env": profile.trust_env,
            "max_retries": profile.max_retries,
            "empty_retry_max_tokens": profile.empty_retry_max_tokens,
        },
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "allow_truncated": allow_truncated,
        "extra": extra,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LLMResponseCache:
    """SQLite-backed、限量且 best-effort 的 LLM 缓存。"""

    def __init__(self, path: Path | None = None, *, max_entries: int | None = None) -> None:
        configured = path or settings.LLM_CACHE_PATH
        self.path = Path(configured or (APP_HOME / "cache" / "llm.sqlite3")).expanduser()
        self.max_entries = (
            settings.LLM_CACHE_MAX_ENTRIES if max_entries is None else max(0, int(max_entries))
        )
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "writes": 0, "errors": 0}

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(**self._stats)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        if self.path.is_symlink():
            raise OSError(f"LLM 缓存不能是符号链接: {self.path}")
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                latency_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS call_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                cache_key TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                latency_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                model TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS responses_accessed_idx ON responses(accessed_at)"
        )
        connection.commit()
        self.path.chmod(0o600)
        return connection

    def record_call(
        self,
        cache_key: str,
        *,
        cache_hit: bool,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        model: str = "",
    ) -> None:
        """记录一次调用摘要，不保存消息正文或凭据。"""

        if not cache_key or self.max_entries <= 0:
            return
        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                connection.execute(
                    """
                    INSERT INTO call_events(
                        created_at, cache_key, cache_hit, latency_ms,
                        prompt_tokens, completion_tokens, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        cache_key,
                        int(cache_hit),
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        model[:300],
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM call_events
                    WHERE event_id NOT IN (
                        SELECT event_id FROM call_events ORDER BY event_id DESC LIMIT 10000
                    )
                    """
                )
                connection.commit()
            except (OSError, sqlite3.Error):
                self._stats["errors"] += 1
            finally:
                if connection is not None:
                    connection.close()

    def get(self, cache_key: str) -> str | None:
        if not cache_key or self.max_entries <= 0:
            return None
        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                row = connection.execute(
                    "SELECT response FROM responses WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if row is None:
                    self._stats["misses"] += 1
                    return None
                connection.execute(
                    "UPDATE responses SET accessed_at = ? WHERE cache_key = ?",
                    (time.time(), cache_key),
                )
                connection.commit()
                self._stats["hits"] += 1
                return str(row[0])
            except (OSError, sqlite3.Error):
                self._stats["errors"] += 1
                return None
            finally:
                if connection is not None:
                    connection.close()

    def set(
        self,
        cache_key: str,
        response: str,
        *,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        if not cache_key or not response or self.max_entries <= 0:
            return
        now = time.time()
        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                connection.execute(
                    """
                    INSERT INTO responses(
                        cache_key, response, created_at, accessed_at,
                        latency_ms, prompt_tokens, completion_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response=excluded.response,
                        accessed_at=excluded.accessed_at,
                        latency_ms=excluded.latency_ms,
                        prompt_tokens=excluded.prompt_tokens,
                        completion_tokens=excluded.completion_tokens
                    """,
                    (
                        cache_key,
                        response,
                        now,
                        now,
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM responses
                    WHERE cache_key IN (
                        SELECT cache_key FROM responses
                        ORDER BY accessed_at DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (self.max_entries,),
                )
                connection.commit()
                self._stats["writes"] += 1
            except (OSError, sqlite3.Error):
                # 缓存是优化项。磁盘只读、锁冲突或损坏不能阻断动画生成。
                self._stats["errors"] += 1
            finally:
                if connection is not None:
                    connection.close()

    def clear(self) -> int:
        """删除全部缓存，返回删除条数；供诊断/维护命令使用。"""

        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                count = int(connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
                connection.execute("DELETE FROM responses")
                connection.execute("DELETE FROM call_events")
                connection.commit()
                return count
            except (OSError, sqlite3.Error):
                self._stats["errors"] += 1
                return 0
            finally:
                if connection is not None:
                    connection.close()

    def summary(self) -> dict[str, object]:
        """返回缓存条目和调用事件计数，不返回任何响应正文。"""

        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                entries = int(connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
                events = int(connection.execute("SELECT COUNT(*) FROM call_events").fetchone()[0])
                return {
                    "path": str(self.path),
                    "entries": entries,
                    "events": events,
                    "stats": asdict(self.stats),
                }
            except (OSError, sqlite3.Error):
                self._stats["errors"] += 1
                return {
                    "path": str(self.path),
                    "entries": 0,
                    "events": 0,
                    "stats": asdict(self.stats),
                }
            finally:
                if connection is not None:
                    connection.close()


def cache_path() -> Path:
    """返回当前配置的缓存路径，不创建文件。"""

    return Path(settings.LLM_CACHE_PATH).expanduser()
