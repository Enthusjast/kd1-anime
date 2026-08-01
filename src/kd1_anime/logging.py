"""结构化日志配置模块。

提供统一的日志配置，支持：
- 按模块/级别过滤
- Rich 格式化输出
- 可选文件日志
- 通过 LLM_DEBUG 控制详细程度
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.logging import RichHandler

# 全局 console 实例，用于 Rich 输出
console = Console()


def setup_logging(
    level: int | str | None = None,
    log_file: Path | None = None,
    debug: bool = False,
) -> None:
    """配置全局日志。
    
    Args:
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 可选的日志文件路径
        debug: 是否启用调试模式（覆盖 level）
    """
    if debug:
        level = logging.DEBUG
    elif level is None:
        level = logging.INFO
    
    # 清除现有 handlers
    root = logging.getLogger()
    root.handlers.clear()
    
    # Rich handler 用于控制台输出
    rich_handler = RichHandler(
        console=console,
        show_path=debug,
        show_time=debug,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)
    
    # 可选的文件 handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)
    
    root.setLevel(logging.DEBUG)  # 根 logger 接收所有，由 handler 过滤


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger。
    
    Args:
        name: logger 名称，通常使用 __name__
        
    Returns:
        配置好的 logger 实例
    """
    return logging.getLogger(name)


# 便捷的模块级 logger
logger = get_logger("kd1_anime")


class AgentLogger:
    """Agent 专用的日志包装器。
    
    提供与原有 console.print 兼容的接口，同时使用标准 logging。
    """
    
    def __init__(self, name: str) -> None:
        self.logger = get_logger(f"kd1_anime.agent.{name}")
        self.name = name
    
    def info(self, message: str, style: str = "bold cyan") -> None:
        """记录信息日志。"""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """记录调试日志。"""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """记录警告日志。"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """记录错误日志。"""
        self.logger.error(message)
    
    def thinking(self, message: str) -> None:
        """记录 Agent 思考过程（INFO 级别）。"""
        self.logger.info(f"[cyan]{self.name}[/] {message}")
    
    def success(self, message: str) -> None:
        """记录成功消息。"""
        self.logger.info(f"[green]✓[/] {message}")
    
    def failure(self, message: str) -> None:
        """记录失败消息。"""
        self.logger.error(f"[red]✗[/] {message}")
    
    def panel(self, title: str, content: str, style: str = "blue") -> None:
        """记录面板内容（用于调试）。"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[{style}]{title}[/]\n{content}")
