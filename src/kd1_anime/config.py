"""全局配置：从用户级配置、当前目录 .env 和系统环境变量加载。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

XDG_CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
USER_CONFIG_DIR = XDG_CONFIG_HOME / "kd1-anime"
USER_ENV_FILE = USER_CONFIG_DIR / ".env"


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
        env_file=(str(USER_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    # --- LLM API ---
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = ""
    LLM_SEND_MAX_TOKENS: bool = True
    LLM_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int | None = Field(default=32768, ge=1, le=1_000_000)
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

    LLM_RETRY_BASE_DELAY: float = Field(default=2.0, ge=0.1, le=120.0)
    LLM_PARALLEL_WORKERS: int = Field(default=4, ge=1, le=16)
    LLM_DEBUG: bool = False
    LLM_USE_JSON_MODE: bool = Field(
        default=True,
        description="是否使用 response_format=json_object。某些端点不支持此参数时会自动降级",
    )

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
    ALLOW_PARTIAL_OUTPUT: bool = False
    OVERWRITE_OUTPUT: bool = False

    # --- 路径 ---
    WORKSPACE_DIR: Path = Path("workspace")
    OUTPUT_FILE: Path = Path("output_final.mp4")

    # 兼容旧调用；实际流水线使用 workspace/runs/<run_id>/ 下的隔离目录。
    SCENES_DIR: Path = Path("workspace/scenes")
    LOGS_DIR: Path = Path("workspace/logs")
    VIDEOS_DIR: Path = Path("workspace/videos")

    # --- Agent 与监控 ---
    MAX_REVIEW_ROUNDS: int = Field(default=5, ge=1, le=10)
    SKIP_REVIEW: bool = Field(default=False, description="是否跳过代码审查阶段")
    # 渲染失败后的最大自动修复次数。autofixer 每轮会调用 LLM 重写代码并重新提交 Slurm。
    MAX_FIX_ATTEMPTS: int = Field(default=5, ge=0, le=20)
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
    EVAL_VISUAL_MODEL: str | None = Field(
        default=None, description="视觉评估使用的模型（默认使用 LLM_MODEL）"
    )
    MAX_SCENES: int = Field(default=12, ge=1, le=100)
    MAX_PROMPT_CHARS: int = Field(default=50_000, ge=100, le=1_000_000)
    MAX_LOG_CHARS: int = Field(default=30_000, ge=1_000, le=1_000_000)
    CODE_VALIDATION_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    MONITOR_POLL_INTERVAL: int = Field(default=10, ge=1)
    MONITOR_QUEUE_TIMEOUT: int = Field(default=3600, ge=1)
    MONITOR_RUN_TIMEOUT: int = Field(default=3600, ge=1)
    MONITOR_MAX_UNKNOWN: int = Field(default=5, ge=1)
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


settings = Settings()
