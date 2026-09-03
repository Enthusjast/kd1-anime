"""把流水线异常映射到可解释、可测试的修复路径。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

FailureCategory = Literal[
    "latex",
    "renderer",
    "ast",
    "lifecycle",
    "math",
    "continuity",
    "infrastructure",
    "merge",
    "render",
    "unknown",
]
RepairHandler = Literal[
    "code_patch",
    "code_rewrite",
    "plan_review",
    "continuity_review",
    "infra_retry",
    "merge_retry",
    "render_fix",
    "manual_diagnosis",
]


@dataclass(frozen=True, slots=True)
class FailureRoute:
    """一次失败的确定性路由结果。"""

    category: FailureCategory
    handler: RepairHandler
    retryable: bool
    reason: str


def classify_failure(
    message: str,
    *,
    phase: str = "",
    status: str = "",
) -> FailureRoute:
    """按最具体的证据选择修复器，不让所有错误都进入 AutoFix。

    分类只依赖文本和已知 Slurm 状态，不执行日志中的内容。顺序很重要：
    缺少工具和资源耗尽优先于普通 LaTeX/运行时错误，避免让 LLM 重写
    本来正确的业务代码。
    """

    text = str(message or "").lower()
    phase_text = str(phase or "").lower()
    status_text = str(status or "").upper()
    infrastructure_markers = (
        "no such file or directory: 'xelatex'",
        "no such file or directory: 'ffmpeg'",
        "no such file or directory: 'ffprobe'",
        "cannot find",
        "invalid account",
        "invalid partition",
        "invalid qos",
        "permission denied",
        "out of memory",
        "time limit",
        "cancelled",
        "cancelled at",
        "node_fail",
        "preempt",
        "unknown_timeout",
        "queue_timeout",
        "run_timeout",
    )
    oom_marker = re.search(r"\boom(?:[- ]kill|\s+event)?\b", text)
    if status_text in {
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
        "CANCELLED",
        "PREEMPTED",
        "QUEUE_TIMEOUT",
        "RUN_TIMEOUT",
        "UNKNOWN_TIMEOUT",
    } or oom_marker is not None or any(marker in text for marker in infrastructure_markers):
        return FailureRoute(
            "infrastructure",
            "infra_retry",
            True,
            "检测到调度器、依赖工具或计算资源问题，禁止优先重写业务代码",
        )
    if phase_text in {"merge", "merging"} or any(
        marker in text for marker in ("ffmpeg", "ffprobe", "concat", "xfade", "合并视频")
    ):
        return FailureRoute("merge", "merge_retry", True, "视频合并或媒体容器处理失败")
    if any(
        marker in text for marker in ("latex", "xelatex", "missing $", "emergency stop", "tex/")
    ):
        return FailureRoute("latex", "code_patch", True, "LaTeX/MathTex 语法或模板问题")
    if any(
        marker in text
        for marker in (
            "should_render",
            "openglcamera",
            "camera.frame",
            "egl",
            "glx",
            "renderer",
        )
    ):
        return FailureRoute("renderer", "code_patch", True, "Manim renderer 或相机 API 不兼容")
    if any(
        marker in text
        for marker in ("active", "transform", "fadeout", "vgroup", "lifecycle", "生命周期")
    ):
        return FailureRoute("lifecycle", "code_patch", True, "Mobject 生命周期或动画源/目标不一致")
    if any(
        marker in text
        for marker in (
            "syntaxerror",
            "nameerror",
            "importerror",
            "typeerror",
            "attributeerror",
            "indexerror",
            "list index out of range",
        )
    ):
        return FailureRoute("ast", "code_patch", True, "Python 语法、名称、参数或 API 调用错误")
    if any(marker in text for marker in ("连续性", "handoff", "inherited", "导出区", "element_id")):
        return FailureRoute("continuity", "continuity_review", True, "场景边界元素或交接合同冲突")
    if any(
        marker in text
        for marker in ("数学", "公式", "不等价", "equation", "eigenvalue", "特征值", "面积")
    ):
        return FailureRoute("math", "plan_review", False, "数学断言或教学计划可能不正确")
    if phase_text in {"render", "monitoring", "fixing"}:
        return FailureRoute("render", "render_fix", True, "渲染运行时错误，交给 AutoFix 分析")
    return FailureRoute("unknown", "manual_diagnosis", False, "没有足够证据确定修复路径")


__all__ = ["FailureCategory", "FailureRoute", "RepairHandler", "classify_failure"]
