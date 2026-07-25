"""
配置管理模块
使用 pydantic-settings 从环境变量或 .env 文件加载配置
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置,支持从环境变量或 .env 文件读取"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM API 配置 (兼容任意 OpenAI 接口) ---
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    # 部分 (推理类) 模型拒绝 max_tokens, 设为 False 则不发送该参数
    LLM_SEND_MAX_TOKENS: bool = True

    # --- Slurm 集群配置 ---
    SLURM_PARTITION: str = "Students"
    SLURM_ACCOUNT: str = ""            # Slurm 账户名 (部分集群强制要求)
    SLURM_QOS: str = "qos_stu_default"
    SLURM_CONDA_ENV: str = "manim_env"
    SLURM_TIME_LIMIT: str = "01:00:00"
    SLURM_CPUS_PER_TASK: int = 4
    SLURM_MEM_GB: str = ""           # 留空则不设置 --mem
    SLURM_GPU_TYPE: str = ""         # 留空表示不申请 GPU; 可选: RTX5090, A100
    SLURM_GPU_COUNT: int = 1         # GPU 数量

    # --- 路径配置 ---
    WORKSPACE_DIR: Path = Path("workspace")
    SCENES_DIR: Path = Path("workspace/scenes")
    LOGS_DIR: Path = Path("workspace/logs")
    VIDEOS_DIR: Path = Path("workspace/videos")
    OUTPUT_FILE: Path = Path("output_final.mp4")

    # --- Agent 配置 ---
    MAX_REVIEW_ROUNDS: int = 3       # Reviewer -> Coder 最大循环次数
    MAX_FIX_ATTEMPTS: int = 3        # Auto-Fix 最大尝试次数
    MAX_CLARIFY_ROUNDS: int = 6      # 需求澄清最大轮次
    MONITOR_POLL_INTERVAL: int = 10  # 轮询 Slurm 状态的间隔 (秒)
    MONITOR_TIMEOUT: int = 3600      # 单个任务最大等待时间 (秒)
    MONITOR_MAX_UNKNOWN: int = 5     # 连续 UNKNOWN 后判定监控失败
    LOG_TAIL_LINES: int = 80         # 读取错误日志的最后 N 行

    # --- LLM 调用配置 ---
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY: float = 2.0  # 指数退避基础延迟 (秒)
    LLM_DEBUG: bool = False            # 开启后在控制台输出 LLM 请求/响应全文

    def require_llm_key(self) -> None:
        """
        快速校验 LLM_API_KEY 已设置.
        在所有入口和 LLM 调用前调用, 避免到深层调用才报错.
        """
        if not self.LLM_API_KEY:
            raise ValueError(
                "未设置 LLM_API_KEY. 请通过以下方式之一设置:\n"
                "  1. .env 文件: LLM_API_KEY=your_key\n"
                "  2. 环境变量: export LLM_API_KEY=your_key\n"
                "  3. 命令行参数: --api-key your_key"
            )


# 全局单例 (即使 .env 缺失也不在此崩溃, 由 require_llm_key 在使用时快速失败)
settings = Settings()
