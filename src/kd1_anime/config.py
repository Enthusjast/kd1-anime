"""全局配置：从用户级配置、当前目录 .env 和系统环境变量加载。"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 应用产生的配置、知识库、缓存和运行产物统一放在一个私有目录中。
# 不跟随 XDG_CONFIG_HOME：集群上不同 shell/module 配置的 XDG 值经常不一致，
# 而单一固定根目录更容易备份、迁移和排查。
APP_HOME = Path.home() / ".kd1-anime"
USER_CONFIG_DIR = APP_HOME
USER_ENV_FILE = APP_HOME / ".env"

# 仅用于从早期版本平滑迁移；新文件永远只写入 APP_HOME。
LEGACY_USER_CONFIG_DIR = Path.home() / ".config" / "kd1-anime"
LEGACY_USER_ENV_FILE = LEGACY_USER_CONFIG_DIR / ".env"

DEFAULT_RAG_INDEX_PATH = APP_HOME / "rag" / "index.sqlite3"
LEGACY_RAG_INDEX_PATH = Path.home() / ".cache" / "kd1-anime" / "rag" / "index.sqlite3"
DEFAULT_KNOWLEDGE_DIR = APP_HOME / "knowledge"
DEFAULT_RAG_DOCS_DIR = DEFAULT_KNOWLEDGE_DIR / "docs"
DEFAULT_RAG_EXAMPLES_DIR = DEFAULT_KNOWLEDGE_DIR / "examples"
DEFAULT_RAG_RECIPES_DIR = DEFAULT_KNOWLEDGE_DIR / "recipes"
DEFAULT_WORKSPACE_DIR = APP_HOME / "workspace"
DEFAULT_SCENES_DIR = DEFAULT_WORKSPACE_DIR / "scenes"
DEFAULT_LOGS_DIR = DEFAULT_WORKSPACE_DIR / "logs"
DEFAULT_VIDEOS_DIR = DEFAULT_WORKSPACE_DIR / "videos"

_LEGACY_STORAGE_DEFAULTS = {
    "RAG_INDEX_PATH": ("~/.cache/kd1-anime/rag/index.sqlite3", str(DEFAULT_RAG_INDEX_PATH)),
    "RAG_DOCS_DIR": ("", str(DEFAULT_RAG_DOCS_DIR)),
    "RAG_EXAMPLES_DIR": ("", str(DEFAULT_RAG_EXAMPLES_DIR)),
    "RAG_RECIPES_DIR": ("", str(DEFAULT_RAG_RECIPES_DIR)),
    "WORKSPACE_DIR": ("workspace", str(DEFAULT_WORKSPACE_DIR)),
    "SCENES_DIR": ("workspace/scenes", str(DEFAULT_SCENES_DIR)),
    "LOGS_DIR": ("workspace/logs", str(DEFAULT_LOGS_DIR)),
    "VIDEOS_DIR": ("workspace/videos", str(DEFAULT_VIDEOS_DIR)),
}


def _rewrite_legacy_storage_defaults(content: str) -> str:
    """只改写旧模板的默认路径，不覆盖用户显式选择的其它路径。"""

    rewritten: list[str] = []
    for line in content.splitlines(keepends=True):
        newline = ""
        body = line
        if body.endswith("\n"):
            body = body[:-1]
            newline = "\n"
            if body.endswith("\r"):
                body = body[:-1]
                newline = "\r\n"
        for key, (legacy, current) in _LEGACY_STORAGE_DEFAULTS.items():
            match = re.match(rf"^(\s*{re.escape(key)}\s*=\s*)(.*)$", body)
            if match is None:
                continue
            raw_value = match.group(2).strip()
            quote = ""
            value = raw_value
            if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "\"'":
                quote = raw_value[0]
                value = raw_value[1:-1]
            if value == legacy:
                body = f"{match.group(1)}{quote}{current}{quote}"
            break
        rewritten.append(body + newline)
    return "".join(rewritten)


def _write_private_text(path: Path, content: str) -> None:
    """以 0600 原子写入迁移后的配置文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # 使用同目录硬链接实现“仅当目标不存在时创建”。相比 os.replace，
        # 并发启动时不会把用户刚创建的新配置覆盖掉。
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def migrate_legacy_user_config(
    legacy_config_dir: Path | None = None,
    user_config_dir: Path | None = None,
) -> tuple[Path, ...]:
    """把旧用户配置非破坏性迁移到 ``~/.kd1-anime``。

    迁移只在目标文件不存在时执行，旧目录和旧文件不会被删除；如果文件系统
    不允许迁移，也由 Settings 的只读旧路径回退继续兼容，不会阻断启动。
    """

    legacy_dir = legacy_config_dir or LEGACY_USER_CONFIG_DIR
    target_dir = user_config_dir or USER_CONFIG_DIR
    if target_dir == legacy_dir:
        return ()
    migrated: list[Path] = []
    for source, destination in (
        (legacy_dir / ".env", target_dir / ".env"),
        (legacy_dir / ".env.example", target_dir / ".env.example"),
    ):
        if destination.exists() or not source.is_file():
            continue
        try:
            content = source.read_text(encoding="utf-8")
            _write_private_text(destination, _rewrite_legacy_storage_defaults(content))
        except (OSError, UnicodeError):
            continue
        migrated.append(destination)
    return tuple(migrated)


# 在构造 Settings 前完成一次非破坏迁移。若目标文件已经存在，则不再把旧文件
# 加入配置源，避免旧配置中的相对 workspace 值补入新配置。
migrate_legacy_user_config()


def _settings_env_files() -> tuple[str, ...]:
    if USER_ENV_FILE.is_file():
        return (str(USER_ENV_FILE), ".env")
    return (str(LEGACY_USER_ENV_FILE), ".env")


@dataclass(frozen=True, slots=True)
class LLMRuntimeProfile:
    """一次 Agent 调用使用的不可变 OpenAI-compatible 端点配置。"""

    label: str
    env_prefix: str
    api_key: str
    base_url: str
    model: str
    send_max_tokens: bool
    temperature: float
    max_tokens: int | None
    max_retries: int
    retry_base_delay: float
    timeout_connect: float
    timeout_read: float
    healthcheck_timeout: float
    silent_stream: bool
    empty_retry_max_tokens: int
    json_repair_attempts: int
    use_json_mode: bool
    debug: bool
    trust_env: bool = True

    def require(self) -> None:
        """验证端点凭据，不把 secret 写入异常文本。"""

        placeholders = {"", "sk-your-key-here", "your-model-name"}
        missing: list[str] = []
        if not self.api_key or self.api_key in placeholders:
            missing.append(f"{self.env_prefix}_API_KEY")
        if not self.base_url.strip():
            missing.append(f"{self.env_prefix}_BASE_URL")
        if not self.model.strip() or self.model in placeholders:
            missing.append(f"{self.env_prefix}_MODEL")
        if missing:
            raise ValueError(f"{self.label}配置不完整（缺少或仍为占位值：{', '.join(missing)}）")


def _current_working_directory() -> Path:
    return Path.cwd()


def resolve_runtime_path(path: Path) -> Path:
    """Resolve configured paths even when the process working directory was removed."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    try:
        base = _current_working_directory()
    except OSError:
        base = Path.home()
    return (base / expanded).resolve()


class Settings(BaseSettings):
    """全局配置；系统环境变量优先于当前目录和用户级 .env。"""

    model_config = SettingsConfigDict(
        # 后面的文件优先级更高，因此项目目录 .env 可覆盖用户级默认配置。
        # 旧文件只作为迁移失败时的只读回退；新配置优先级高于旧配置。
        env_file=_settings_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    # --- LLM API ---
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = ""
    # 可选的阶段级模型路由；为空时回退到 LLM_MODEL。端点和密钥仍由
    # 主模型配置统一管理，避免不同阶段意外使用不同凭据。
    LLM_PLANNING_MODEL: str = ""
    LLM_TECHNICAL_MODEL: str = ""
    LLM_CODE_MODEL: str = ""
    LLM_REVIEW_MODEL: str = ""
    LLM_FIX_MODEL: str = ""
    LLM_SEND_MAX_TOKENS: bool = True
    LLM_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    # 阶段级温度：结构化合同保持确定性，代码创作保留少量探索空间。
    LLM_PLANNING_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_TECHNICAL_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_CODE_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_REVIEW_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_FIX_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int | None = Field(default=32768, ge=1, le=1_000_000)
    # 不同阶段的输出复杂度差异很大。默认使用较小的阶段预算，避免计划
    # 审查/连续性审查为极短 JSON 消耗与代码生成相同的长推理预算；用户
    # 仍可按模型能力覆盖这些值，LLM_MAX_TOKENS 保留为总配置兼容项。
    LLM_PLANNING_MAX_TOKENS: int = Field(default=16384, ge=4096, le=1_000_000)
    LLM_TECHNICAL_MAX_TOKENS: int = Field(default=16384, ge=4096, le=1_000_000)
    LLM_CODE_MAX_TOKENS: int = Field(default=24576, ge=4096, le=1_000_000)
    LLM_REVIEW_MAX_TOKENS: int = Field(default=8192, ge=2048, le=1_000_000)
    LLM_MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    # 单次 LLM 请求的连接/读取超时(秒)。读取超时对非流式是"等待完整响应"，
    # 对流式是"等待下一个 chunk"——静默流式下 600s 只是兜底，不会拖慢任何请求。
    LLM_TIMEOUT_CONNECT: float = Field(default=30.0, ge=1.0, le=300.0)
    LLM_TIMEOUT_READ: float = Field(default=600.0, ge=10.0, le=3600.0)
    # 非流式(stream=False)调用是否仍使用流式传输但静默收集(不打印内容)。
    # 推理模型(reasoning_content)在完整响应生成完之前不会返回任何字节，
    # 非流式 + 短读超时会导致"超时→重头生成"的级联；静默流式按 chunk 计超时，
    # 内容一开始生成就能收到，终端表现与非流式完全一致。
    LLM_SILENT_STREAM: bool = Field(
        default=True, description="stream=False 时仍走流式传输但静默收集，避免长生成超时"
    )
    # CLI 启动时的最小可用性探测超时；不能复用完整生成请求的长读取超时，
    # 否则端点不可用时用户需要等待数分钟才看到明确错误。
    LLM_HEALTHCHECK_TIMEOUT: float = Field(
        default=15.0, ge=1.0, le=120.0, description="CLI 启动时 LLM 可用性探测超时（秒）"
    )
    # 空响应重试时补上的 max_tokens 兜底值：推理模型常把输出预算耗尽在思考上，
    # 导致 content 为空；补足预算后重试可避免反复拿到空响应。
    LLM_EMPTY_RETRY_MAX_TOKENS: int = Field(default=16384, ge=1024, le=65536)
    # 结构化 JSON 输出未通过 Pydantic 校验时, 带错误反馈重试的次数 (0=关闭)。
    # 模型偶尔会返回不合规的枚举值/缺字段 (如 severity="none"), 直接判死整个
    # 场景太浪费; 把校验错误喂回模型重试, 通常一次即可修正。
    LLM_JSON_REPAIR_ATTEMPTS: int = Field(default=2, ge=0, le=5)

    @field_validator("LLM_MAX_TOKENS", mode="before")
    @classmethod
    def validate_max_tokens(cls, value):
        if value is None or value == "":
            return None
        return value

    @field_validator("LLM_BASE_URL")
    @classmethod
    def validate_llm_base_url(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("LLM_BASE_URL 不能为空")
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM_BASE_URL 必须是带 http/https scheme 的 URL")
        if parsed.username or parsed.password:
            raise ValueError("LLM_BASE_URL 不能把凭据写在 URL 中，请使用 API_KEY")
        if parsed.fragment:
            raise ValueError("LLM_BASE_URL 不能包含 URL fragment")
        if any(char in value for char in "\r\n"):
            raise ValueError("LLM_BASE_URL 不能包含换行")
        return value.strip().rstrip("/")

    LLM_RETRY_BASE_DELAY: float = Field(default=2.0, ge=0.1, le=120.0)
    LLM_PARALLEL_WORKERS: int = Field(default=4, ge=1, le=16)
    LLM_DEBUG: bool = False
    LLM_TRUST_ENV: bool = Field(
        default=True,
        description="是否让 HTTP 客户端读取 HTTP(S)_PROXY 等环境变量",
    )
    LLM_USE_JSON_MODE: bool = Field(
        default=True,
        description="是否使用 response_format=json_object。某些端点不支持此参数时会自动降级",
    )
    # 非流式业务请求的本地响应缓存。缓存只保存去除 API Key 后的请求指纹和
    # 完整文本响应，默认位于用户私有目录；流式交互请求永不写入缓存。
    LLM_CACHE_ENABLED: bool = Field(
        default=True,
        description="是否启用本地 LLM 响应缓存（不缓存流式交互请求）",
    )
    LLM_CACHE_PATH: Path = APP_HOME / "cache" / "llm.sqlite3"
    LLM_CACHE_MAX_ENTRIES: int = Field(default=512, ge=0, le=100_000)
    # 各 Agent 的 user message 统一使用区块预算；代码和结构化合同不会被
    # 裁剪，低优先级的 RAG/自然语言说明会优先让出空间。
    LLM_MAX_CONTEXT_CHARS: int = Field(default=120_000, ge=10_000, le=2_000_000)
    LLM_MAX_CODE_CONTEXT_CHARS: int = Field(default=60_000, ge=5_000, le=1_000_000)
    LLM_MAX_REVIEW_CONTEXT_CHARS: int = Field(default=90_000, ge=10_000, le=2_000_000)
    LLM_MAX_TECHNICAL_SPEC_CHARS: int = Field(default=30_000, ge=5_000, le=500_000)
    MAX_TECHNICAL_SPEC_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    CODEGEN_MODE: Literal["hybrid", "python", "ir"] = Field(
        default="hybrid",
        description="代码生成模式：普通 Python、结构化 IR，或失败时 IR 回退",
    )

    # --- 独立视觉 LLM API ---
    # 视觉评估绝不隐式复用主 LLM 的 Key、端点或模型。未启用时可以留空。
    VISUAL_LLM_API_KEY: str = ""
    VISUAL_LLM_BASE_URL: str = ""
    VISUAL_LLM_MODEL: str = ""
    VISUAL_LLM_SEND_MAX_TOKENS: bool = True
    VISUAL_LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    VISUAL_LLM_MAX_TOKENS: int | None = Field(default=3000, ge=1, le=1_000_000)
    VISUAL_LLM_MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    VISUAL_LLM_RETRY_BASE_DELAY: float = Field(default=2.0, ge=0.1, le=120.0)
    VISUAL_LLM_TIMEOUT_CONNECT: float = Field(default=30.0, ge=1.0, le=300.0)
    VISUAL_LLM_TIMEOUT_READ: float = Field(default=300.0, ge=10.0, le=3600.0)
    VISUAL_LLM_HEALTHCHECK_TIMEOUT: float = Field(default=20.0, ge=1.0, le=120.0)
    VISUAL_LLM_JSON_REPAIR_ATTEMPTS: int = Field(default=1, ge=0, le=5)
    VISUAL_LLM_USE_JSON_MODE: bool = True
    VISUAL_LLM_PARALLEL_WORKERS: int = Field(default=2, ge=1, le=16)
    VISUAL_LLM_DEBUG: bool = False
    VISUAL_LLM_TRUST_ENV: bool = Field(
        default=True,
        description="视觉 LLM 是否读取 HTTP(S)_PROXY 等环境变量",
    )

    @field_validator("VISUAL_LLM_BASE_URL")
    @classmethod
    def validate_visual_llm_base_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("VISUAL_LLM_BASE_URL 必须是字符串")
        value = value.strip()
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("VISUAL_LLM_BASE_URL 必须是带 http/https scheme 的 URL")
        if parsed.username or parsed.password:
            raise ValueError("VISUAL_LLM_BASE_URL 不能把凭据写在 URL 中，请使用 API_KEY")
        if parsed.fragment:
            raise ValueError("VISUAL_LLM_BASE_URL 不能包含 URL fragment")
        if any(char in value for char in "\r\n"):
            raise ValueError("VISUAL_LLM_BASE_URL 不能包含换行")
        return value.rstrip("/")

    @field_validator("VISUAL_LLM_MAX_TOKENS", mode="before")
    @classmethod
    def validate_visual_max_tokens(cls, value):
        if value is None or value == "":
            return None
        return value

    # --- 独立 RAG 服务 ---
    # RAG 的 Embedding/Reranker 端点和主/视觉 LLM 完全隔离；默认关闭，
    # 因而空 URL 不会影响普通动画生成。
    RAG_ENABLED: bool = False
    RAG_INDEX_PATH: Path = DEFAULT_RAG_INDEX_PATH
    # 知识库源文件也有固定的用户目录；用户仍可用绝对路径接入其它文档。
    RAG_DOCS_DIR: Path | None = DEFAULT_RAG_DOCS_DIR
    RAG_EXAMPLES_DIR: Path | None = DEFAULT_RAG_EXAMPLES_DIR
    RAG_RECIPES_DIR: Path | None = DEFAULT_RAG_RECIPES_DIR
    RAG_EMBEDDING_API_KEY: str = ""
    RAG_EMBEDDING_BASE_URL: str = ""
    RAG_EMBEDDING_MODEL: str = ""
    RAG_EMBEDDING_TIMEOUT: float = Field(default=60.0, ge=1.0, le=3_600.0)
    RAG_EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    RAG_RERANK_API_KEY: str = ""
    RAG_RERANK_BASE_URL: str = ""
    RAG_RERANK_MODEL: str = ""
    RAG_RERANK_TIMEOUT: float = Field(default=60.0, ge=1.0, le=3_600.0)
    RAG_TRUST_ENV: bool = Field(
        default=True,
        description="RAG HTTP 客户端是否读取 HTTP(S)_PROXY 等环境变量",
    )
    RAG_TOP_K: int = Field(default=8, ge=1, le=100)
    RAG_RERANK_TOP_N: int = Field(default=4, ge=1, le=100)
    RAG_MAX_CONTEXT_CHARS: int = Field(default=12_000, ge=500, le=50_000)
    RAG_CHUNK_SIZE: int = Field(default=1_800, ge=100, le=100_000)
    RAG_CHUNK_OVERLAP: int = Field(default=200, ge=0, le=20_000)
    RAG_PARALLEL_WORKERS: int = Field(default=2, ge=1, le=16)

    @field_validator("RAG_DOCS_DIR", "RAG_EXAMPLES_DIR", "RAG_RECIPES_DIR", mode="before")
    @classmethod
    def normalize_rag_source_dir(cls, value):
        """空的 RAG 源目录必须保持为 None，不能变成 Path('.')。"""

        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("RAG_INDEX_PATH", mode="before")
    @classmethod
    def normalize_rag_index_path(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_RAG_INDEX_PATH
        return value

    @field_validator("RAG_EMBEDDING_BASE_URL", "RAG_RERANK_BASE_URL")
    @classmethod
    def validate_rag_base_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("RAG 服务 URL 必须是字符串")
        value = value.strip()
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RAG 服务 URL 必须是带 http/https scheme 的 URL")
        if parsed.username or parsed.password:
            raise ValueError("RAG 服务 URL 不能把凭据写在 URL 中，请使用独立 API_KEY")
        if parsed.fragment:
            raise ValueError("RAG 服务 URL 不能包含 URL fragment")
        if any(char in value for char in "\r\n"):
            raise ValueError("RAG 服务 URL 不能包含换行")
        return value.rstrip("/")

    @field_validator("RAG_CHUNK_OVERLAP")
    @classmethod
    def validate_rag_chunk_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("RAG_CHUNK_SIZE", 1_800)
        if value >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP 必须小于 RAG_CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def validate_rag_settings(self) -> Settings:
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError("RAG_CHUNK_OVERLAP 必须小于 RAG_CHUNK_SIZE")
        return self

    # --- Slurm 集群 ---
    SLURM_PARTITION: str = ""
    SLURM_ACCOUNT: str = ""
    SLURM_QOS: str = ""
    SLURM_CONDA_ENV: str = "manim_env"
    SLURM_CONDA_BASE: Path | None = None
    SLURM_TIME_LIMIT: str = "01:00:00"
    SLURM_CPUS_PER_TASK: int = Field(default=4, ge=1)
    SLURM_MEM_GB: str = ""
    SLURM_GPU_TYPE: str = ""
    SLURM_GPU_COUNT: int = Field(default=1, ge=1)
    AUTO_RESOURCE_ESTIMATION: bool = Field(
        default=False,
        description="是否按场景复杂度自动增加 Slurm CPU/内存/时间资源",
    )
    # 0 表示不额外限制；正整数用于避免一次向共享集群提交过多场景。
    SLURM_MAX_IN_FLIGHT: int = Field(default=0, ge=0, le=1_000)
    SLURM_SUBMIT_RETRIES: int = Field(default=3, ge=1, le=10)
    SLURM_SUBMIT_RETRY_DELAY: float = Field(default=2.0, ge=0.1, le=120.0)
    SLURM_CONTAINER_IMAGE: Path | None = None
    SLURM_REQUIRE_CONTAINER: bool = False
    SLURM_CONTAINER_DISABLE_NETWORK: bool = Field(
        default=False,
        description="在 Apptainer 支持时为生成代码禁用容器网络",
    )

    # --- Manim 渲染 ---
    MANIM_RENDERER: Literal["cairo", "opengl"] = "cairo"
    MANIM_QUALITY: Literal["l", "m", "h", "p", "k"] = "h"
    MANIM_PIXEL_WIDTH: int = Field(default=1920, ge=16, multiple_of=2)
    MANIM_PIXEL_HEIGHT: int = Field(default=1080, ge=16, multiple_of=2)
    MANIM_FRAME_RATE: int = Field(default=60, ge=1, le=240)
    MANIM_OPENGL_PLATFORM: Literal["egl", "glx"] = "egl"
    # 正式 Slurm 渲染前先用同一 renderer/节点资源执行一次轻量 smoke render，
    # 及早暴露 OpenGL、XeLaTeX、Manim API 和运行时异常；dry-run 永不执行它。
    SMOKE_RENDER_ENABLED: bool = Field(
        default=True,
        description="正式渲染前是否执行轻量 Smoke Render",
    )
    SMOKE_RENDER_MODE: Literal["frame", "video", "both"] = Field(
        default="frame",
        description="Smoke Render 模式：最后一帧、MP4 或两者",
    )
    SMOKE_RENDER_QUALITY: Literal["l", "m"] = "l"
    SMOKE_RENDER_TIMEOUT: int = Field(default=180, ge=10, le=3_600)
    # 本地生成/无 Slurm 环境的可选运行时预检；默认关闭，避免在 dry-run
    # 或共享登录节点上执行不可信生成代码。
    LOCAL_SMOKE_RENDER_ENABLED: bool = False
    LOCAL_SMOKE_RENDER_MODE: Literal["frame", "video", "both"] = "frame"
    LOCAL_SMOKE_RENDER_QUALITY: Literal["l", "m"] = "l"
    LOCAL_SMOKE_RENDER_TIMEOUT: int = Field(default=180, ge=10, le=3_600)
    LOCAL_SMOKE_RENDER_MEMORY_MB: int = Field(default=4_096, ge=256, le=65_536)
    ALLOW_PARTIAL_OUTPUT: bool = False
    OVERWRITE_OUTPUT: bool = False
    TRANSITION_TYPE: Literal["fade"] = "fade"
    TRANSITION_DURATION: float = Field(default=0.5, gt=0.0, le=5.0)
    # --- FFmpeg 合并 ---
    # 合并配置与 Manim RenderProfile 分离：修改这些项不会让已有场景视频
    # 被误认为可以复用，但会在最终合并时明确使用用户选择的发布配置。
    MERGE_VIDEO_CODEC: Literal["libx264", "libx265"] = "libx264"
    MERGE_VIDEO_PRESET: str = Field(
        default="medium",
        pattern=r"^[A-Za-z0-9_-]{1,30}$",
        description="FFmpeg 编码 preset，限制为单行安全标识",
    )
    MERGE_VIDEO_CRF: int = Field(default=18, ge=0, le=51)
    MERGE_AUDIO_SAMPLE_RATE: int = Field(default=48_000, ge=8_000, le=192_000)
    MERGE_AUDIO_CHANNEL_LAYOUT: Literal["stereo"] = "stereo"

    # --- 路径 ---
    WORKSPACE_DIR: Path = DEFAULT_WORKSPACE_DIR
    OUTPUT_FILE: Path = Path("output_final.mp4")

    # 兼容旧调用；实际流水线使用 ~/.kd1-anime/workspace/runs/<run_id>/ 下的隔离目录。
    SCENES_DIR: Path = DEFAULT_SCENES_DIR
    LOGS_DIR: Path = DEFAULT_LOGS_DIR
    VIDEOS_DIR: Path = DEFAULT_VIDEOS_DIR

    # --- Agent 与监控 ---
    MAX_REVIEW_ROUNDS: int = Field(default=5, ge=1, le=10)
    MAX_LOW_RISK_REVIEW_ROUNDS: int = Field(
        default=2,
        ge=1,
        le=10,
        description="低风险场景的语义代码审查轮数；确定性检查始终执行",
    )
    # 单个场景计划在进入 Coder 前允许的重规划审查轮数。
    MAX_PLAN_REVIEW_ROUNDS: int = Field(default=2, ge=1, le=10)
    # 同一场景在计划审查反馈后允许重新调用 Planner 的总次数；独立于
    # 单份计划内部的审查轮数，防止“重规划后计数归零”造成无限循环。
    MAX_PLAN_REPLAN_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    # 全片分镜连续性审查发现冲突后的最大局部重规划轮数。
    MAX_CONTINUITY_FIX_ROUNDS: int = Field(default=2, ge=0, le=10)
    CONTINUITY_CONTEXT_MODE: Literal["minimal", "full", "stateless"] = Field(
        default="minimal",
        description="Coder 接收的跨场景上下文范围；默认只传递当前场景需要的元素",
    )
    SKIP_REVIEW: bool = Field(default=False, description="是否跳过代码审查阶段")
    SAFE_FALLBACK_ENABLED: bool = Field(
        default=True,
        description="复杂几何方案审查耗尽后是否自动切换为保守教学方案",
    )
    MAX_IDENTICAL_REVIEW_ATTEMPTS: int = Field(
        default=2,
        ge=2,
        le=5,
        description="相同代码与相同审查反馈连续出现多少次后提前终止",
    )
    MAX_STAGNANT_ATTEMPTS: int = Field(
        default=2,
        ge=1,
        le=10,
        description="渲染修复没有改变代码或错误指纹多少次后切换确定性回退",
    )
    # 渲染失败后的最大自动修复次数。autofixer 每轮会调用 LLM 重写代码并重新提交 Slurm。
    MAX_FIX_ATTEMPTS: int = Field(default=5, ge=0, le=20)
    # Slurm 节点故障/抢占等与代码无关的终态，允许自动重新排队的次数。
    MAX_INFRA_RETRIES: int = Field(default=2, ge=0, le=10)
    # 连续 N 次渲染错误日志指纹相同 → 提前放弃, 避免 LLM 反复"修复"同一个
    # 环境错误浪费尝试次数。注意该检查在 _scene_fix 中还要叠加 fix_attempts>=2
    # 门槛, 确保修复器至少有 2 次真实尝试, 不会因一次修复失败就误判放弃。
    MAX_FIX_IDENTICAL_ERRORS: int = Field(default=3, ge=1, le=10)
    MAX_CLARIFY_ROUNDS: int = Field(default=12, ge=1, le=20)

    # --- 自动评估配置 ---
    ENABLE_AUTO_EVAL: bool = Field(default=False, description="是否启用自动评估-改进循环")
    ENABLE_VISUAL_EVAL: bool = Field(
        default=False, description="是否启用视觉效果评估（需要 LLM 支持多模态）"
    )
    EVAL_THRESHOLD: float = Field(
        default=3.5, ge=1.0, le=5.0, description="评估通过阈值（1-5分），低于此分数触发改进"
    )
    MAX_EVAL_ROUNDS: int = Field(default=2, ge=0, le=5, description="最大评估-改进轮数")
    EVAL_VISUAL_MODEL: str | None = Field(default=None, description="VISUAL_LLM_MODEL 的弃用别名")
    VISUAL_EVAL_FRAME_COUNT: int = Field(default=6, ge=1, le=8)
    VISUAL_EVAL_THRESHOLD: float = Field(default=3.5, ge=1.0, le=5.0)
    MAX_VISUAL_FIX_ATTEMPTS: int = Field(default=2, ge=0, le=5)
    MAX_SCENES: int = Field(default=12, ge=1, le=100)
    MAX_PROMPT_CHARS: int = Field(default=50_000, ge=100, le=1_000_000)
    # 澄清对话会携带多轮 user/assistant 消息；独立预算避免累计内容超过模型上下文。
    MAX_CLARIFY_CONTEXT_CHARS: int = Field(default=40_000, ge=2_000, le=1_000_000)
    MAX_LOG_CHARS: int = Field(default=30_000, ge=1_000, le=1_000_000)
    CODE_VALIDATION_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    MAX_CODE_CANDIDATES_LOW: int = Field(
        default=1,
        ge=1,
        le=5,
        description="低风险场景允许的不同代码策略数",
    )
    MAX_CODE_CANDIDATES_MEDIUM: int = Field(
        default=2,
        ge=1,
        le=5,
        description="中风险场景允许的不同代码策略数",
    )
    MAX_CODE_CANDIDATES_HIGH: int = Field(
        default=3,
        ge=1,
        le=5,
        description="高风险场景允许的不同代码策略数",
    )
    MONITOR_POLL_INTERVAL: int = Field(default=10, ge=1)
    MONITOR_QUEUE_TIMEOUT: int = Field(default=3600, ge=1)
    MONITOR_RUN_TIMEOUT: int = Field(default=3600, ge=1)
    MONITOR_MAX_UNKNOWN: int = Field(default=5, ge=1)
    # 集群控制面暂时不可查询时，至少连续达到 MONITOR_MAX_UNKNOWN 次且持续
    # 超过此时间才取消作业；避免短暂 squeue/sacct 故障误杀远端任务。
    MONITOR_UNKNOWN_TIMEOUT: int = Field(default=300, ge=1)
    # Slurm 报告 COMPLETED 后，等待共享文件系统传播最终 MP4 的宽限时间。
    MONITOR_ARTIFACT_GRACE: int = Field(default=60, ge=0)
    LOG_TAIL_LINES: int = Field(default=80, ge=1)

    # 旧配置兼容项；若用户仍设置 MONITOR_TIMEOUT，Slurm 层会将其作为显式 override。
    MONITOR_TIMEOUT: int | None = Field(default=None, ge=1)

    @field_validator(
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_QOS",
        "SLURM_CONDA_ENV",
        "SLURM_GPU_TYPE",
    )
    @classmethod
    def validate_slurm_identifier(cls, value: str) -> str:
        """拒绝换行和 shell/Slurm 指令注入，只保留常见安全标识字符。"""

        if not value:
            return value
        if not re.fullmatch(r"[A-Za-z0-9_.:@,+/-]+", value):
            raise ValueError("只能包含字母、数字及 _ . : @ , + / -")
        return value

    @field_validator("SLURM_CONTAINER_IMAGE", mode="before")
    @classmethod
    def normalize_container_image(cls, value):
        """把 .env 中留空/纯空白的 SLURM_CONTAINER_IMAGE 视为未配置。

        否则 Path("") 会解析为 Path(".")（truthy），渲染脚本会误判为
        "Apptainer 镜像不存在: <当前目录>"。
        """

        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("LLM_CACHE_PATH", mode="before")
    @classmethod
    def normalize_llm_cache_path(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return APP_HOME / "cache" / "llm.sqlite3"
        return value

    @field_validator("SLURM_CONDA_BASE", mode="before")
    @classmethod
    def normalize_conda_base(cls, value):
        """把空的 Conda 根目录配置视为未配置。

        ``Path("")`` 会变成当前目录 ``Path('.')``，这会让渲染脚本把空的
        ``SLURM_CONDA_BASE=`` 误当成有效路径，并在远端生成错误的激活命令。
        """

        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("SLURM_TIME_LIMIT")
    @classmethod
    def validate_slurm_time_limit(cls, value: str) -> str:
        match = re.fullmatch(r"(?:\d+-)?(\d{1,3}):(\d{2}):(\d{2})", value)
        if not match:
            raise ValueError("必须使用 [days-]HH:MM:SS 格式")
        if int(match.group(2)) >= 60 or int(match.group(3)) >= 60:
            raise ValueError("MM 和 SS 必须小于 60")
        return value

    @field_validator("SLURM_MEM_GB")
    @classmethod
    def validate_slurm_memory(cls, value: str) -> str:
        if value and not re.fullmatch(r"\d+(?:[KMGTP])?", value, flags=re.IGNORECASE):
            raise ValueError("必须是整数或带 K/M/G/T/P 单位的内存值，例如 32G")
        return value

    @property
    def user_config_dir(self) -> Path:
        return USER_CONFIG_DIR

    @property
    def user_env_file(self) -> Path:
        return USER_ENV_FILE

    def require_llm_key(self) -> None:
        """兼容旧方法名：验证调用 OpenAI-compatible API 所需的完整配置。"""
        missing: list[str] = []
        placeholder_values = {"sk-your-key-here", "your-model-name", ""}

        if not self.LLM_API_KEY or self.LLM_API_KEY in placeholder_values:
            missing.append("LLM_API_KEY")
        if not self.LLM_BASE_URL.strip():
            missing.append("LLM_BASE_URL")
        if not self.LLM_MODEL.strip() or self.LLM_MODEL in placeholder_values:
            missing.append("LLM_MODEL")

        if missing:
            config_path = self.user_env_file
            example_path = Path.cwd() / ".env.example"

            error_msg = f"""LLM 配置不完整（缺少或仍为占位值：{", ".join(missing)}）

配置方法（按优先级）：
1. 设置环境变量：
   export LLM_API_KEY=your-api-key
   export LLM_BASE_URL=your-api-url
   export LLM_MODEL=your-model-name

2. 编辑配置文件：
   {config_path}

3. 在项目目录创建 .env：
   {Path.cwd() / ".env"}

配置示例见：{example_path}
            """
            raise ValueError(error_msg)

    def require_visual_llm(self) -> None:
        """验证独立多模态端点；禁止静默回退到主 LLM。"""

        self.visual_llm_profile().require()

    def main_llm_profile(self, *, stage: str = "default") -> LLMRuntimeProfile:
        """构造主 Agent 配置，并按阶段选择可选模型。"""

        stage_key = str(stage or "default").strip().lower()
        model_fields = {
            "planning": "LLM_PLANNING_MODEL",
            "technical": "LLM_TECHNICAL_MODEL",
            "code": "LLM_CODE_MODEL",
            "review": "LLM_REVIEW_MODEL",
            "fix": "LLM_FIX_MODEL",
        }
        temperature_fields = {
            "planning": "LLM_PLANNING_TEMPERATURE",
            "technical": "LLM_TECHNICAL_TEMPERATURE",
            "code": "LLM_CODE_TEMPERATURE",
            "review": "LLM_REVIEW_TEMPERATURE",
            "fix": "LLM_FIX_TEMPERATURE",
        }
        token_fields = {
            "planning": "LLM_PLANNING_MAX_TOKENS",
            "technical": "LLM_TECHNICAL_MAX_TOKENS",
            "code": "LLM_CODE_MAX_TOKENS",
            "review": "LLM_REVIEW_MAX_TOKENS",
        }
        model = getattr(self, model_fields.get(stage_key, ""), "") or self.LLM_MODEL
        temperature = getattr(
            self,
            temperature_fields.get(stage_key, "LLM_TEMPERATURE"),
            self.LLM_TEMPERATURE,
        )
        max_tokens = getattr(
            self,
            token_fields.get(stage_key, "LLM_MAX_TOKENS"),
            self.LLM_MAX_TOKENS,
        )
        if stage_key == "fix" and not getattr(self, "LLM_FIX_MODEL", ""):
            max_tokens = self.LLM_CODE_MAX_TOKENS

        return LLMRuntimeProfile(
            label="LLM ",
            env_prefix="LLM",
            api_key=self.LLM_API_KEY,
            base_url=self.LLM_BASE_URL,
            model=model,
            send_max_tokens=self.LLM_SEND_MAX_TOKENS,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=self.LLM_MAX_RETRIES,
            retry_base_delay=self.LLM_RETRY_BASE_DELAY,
            timeout_connect=self.LLM_TIMEOUT_CONNECT,
            timeout_read=self.LLM_TIMEOUT_READ,
            healthcheck_timeout=self.LLM_HEALTHCHECK_TIMEOUT,
            silent_stream=self.LLM_SILENT_STREAM,
            empty_retry_max_tokens=self.LLM_EMPTY_RETRY_MAX_TOKENS,
            json_repair_attempts=self.LLM_JSON_REPAIR_ATTEMPTS,
            use_json_mode=self.LLM_USE_JSON_MODE,
            debug=self.LLM_DEBUG,
            trust_env=self.LLM_TRUST_ENV,
        )

    def visual_llm_profile(self, *, model_override: str | None = None) -> LLMRuntimeProfile:
        """构造视觉评估模型配置；只兼容旧模型名，不继承主 API。"""

        model = model_override or self.VISUAL_LLM_MODEL or self.EVAL_VISUAL_MODEL or ""
        return LLMRuntimeProfile(
            label="视觉 LLM ",
            env_prefix="VISUAL_LLM",
            api_key=self.VISUAL_LLM_API_KEY,
            base_url=self.VISUAL_LLM_BASE_URL,
            model=model,
            send_max_tokens=self.VISUAL_LLM_SEND_MAX_TOKENS,
            temperature=self.VISUAL_LLM_TEMPERATURE,
            max_tokens=self.VISUAL_LLM_MAX_TOKENS,
            max_retries=self.VISUAL_LLM_MAX_RETRIES,
            retry_base_delay=self.VISUAL_LLM_RETRY_BASE_DELAY,
            timeout_connect=self.VISUAL_LLM_TIMEOUT_CONNECT,
            timeout_read=self.VISUAL_LLM_TIMEOUT_READ,
            healthcheck_timeout=self.VISUAL_LLM_HEALTHCHECK_TIMEOUT,
            # 图片消息保持普通非流式，兼容更多 OpenAI-compatible 端点。
            silent_stream=False,
            empty_retry_max_tokens=max(1024, self.VISUAL_LLM_MAX_TOKENS or 3000),
            json_repair_attempts=self.VISUAL_LLM_JSON_REPAIR_ATTEMPTS,
            use_json_mode=self.VISUAL_LLM_USE_JSON_MODE,
            debug=self.VISUAL_LLM_DEBUG,
            trust_env=self.VISUAL_LLM_TRUST_ENV,
        )


settings = Settings()
