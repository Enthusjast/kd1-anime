"""项目自定义异常层次结构。

提供细粒度的异常类型，便于精确捕获和处理不同错误场景。
"""

from __future__ import annotations


class KD1Error(Exception):
    """所有项目异常的基类。"""

    def __init__(self, message: str = "", *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class ConfigError(KD1Error):
    """配置相关错误（缺失、无效、冲突）。"""


class LLMError(KD1Error):
    """LLM API 调用相关错误。"""


class LLMAuthError(LLMError):
    """LLM API 认证/授权错误（API key 无效、过期）。"""


class LLMRateLimitError(LLMError):
    """LLM API 速率限制错误。"""


class LLMTimeoutError(LLMError):
    """LLM API 超时错误。"""


class LLMResponseError(LLMError):
    """LLM 响应格式/内容错误（无法解析、不符合 schema）。"""


class SlurmError(KD1Error):
    """Slurm 调度相关错误。"""


class SlurmSubmitError(SlurmError):
    """sbatch 提交失败。"""


class SlurmCancelError(SlurmError):
    """scancel 取消失败。"""


class SlurmTimeoutError(SlurmError):
    """Slurm 作业超时（排队或运行）。"""


class SlurmResourceError(SlurmError):
    """Slurm 资源配置错误（GPU 类型未配置、分区无效等）。"""


class RenderError(KD1Error):
    """Manim 渲染相关错误。"""


class RenderTimeoutError(RenderError):
    """渲染超时。"""


class RenderOOMError(RenderError):
    """渲染内存不足。"""


class ValidationError(KD1Error):
    """代码校验相关错误。"""


class CodeValidationError(ValidationError):
    """生成的代码未通过 AST 校验。"""


class ReviewError(ValidationError):
    """代码审查发现严重问题。"""


class MediaError(KD1Error):
    """视频/媒体处理相关错误。"""


class FFmpegError(MediaError):
    """FFmpeg 执行失败。"""


class VideoNotFoundError(MediaError):
    """未找到渲染后的视频文件。"""


class MergeError(MediaError):
    """视频合并失败。"""


class RunError(KD1Error):
    """运行管理相关错误。"""


class RunNotFoundError(RunError):
    """指定的运行 ID 不存在。"""


class RunLockError(RunError):
    """无法获取运行锁（可能有其他进程正在使用）。"""


class RunIntegrityError(RunError):
    """运行数据完整性校验失败（代码 hash 不匹配、路径逃逸）。"""


class PipelineError(KD1Error):
    """流水线执行相关错误。"""


class PipelineAbortedError(PipelineError):
    """流水线被用户中断。"""


class PipelineExhaustedError(PipelineError):
    """重试/修复次数耗尽。"""


class InstallError(KD1Error):
    """安装/环境配置相关错误。"""


class DependencyError(InstallError):
    """缺少必要的依赖（conda、ffmpeg、texlive 等）。"""


class EnvironmentConfigError(InstallError):
    """环境配置错误（conda 环境损坏、路径问题等）。

    注意：原名 EnvironmentError 会遮蔽内置异常，已重命名为 EnvironmentConfigError。
    """
