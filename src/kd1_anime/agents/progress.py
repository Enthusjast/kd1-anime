"""用于判断代码修复是否真正取得进展的小型状态模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

ProgressKind = Literal["improved", "unchanged", "regressed", "unknown"]


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """一次候选代码/运行结果的稳定摘要。"""

    code_sha256: str
    error_fingerprint: str = ""
    issue_count: int | None = None

    @classmethod
    def from_values(
        cls,
        code: str,
        *,
        error_fingerprint: str = "",
        issue_count: int | None = None,
    ) -> ProgressSnapshot:
        return cls(
            code_sha256=(hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""),
            error_fingerprint=str(error_fingerprint or ""),
            issue_count=issue_count,
        )


def classify_progress(
    previous: ProgressSnapshot | None,
    current: ProgressSnapshot,
) -> ProgressKind:
    """比较相邻尝试，避免同一候选在修复循环中无限重放。

    错误指纹发生变化表示运行行为已经改变，即使代码暂时没有变化也属于
    改善。若调用方提供确定性问题数量，问题变多则明确标记为退化；没有
    这些证据时不会凭主观偏好判定“退化”。
    """

    if previous is None or not previous.code_sha256:
        return "unknown"
    if (
        previous.issue_count is not None
        and current.issue_count is not None
        and current.issue_count > previous.issue_count
    ):
        return "regressed"
    if (
        previous.code_sha256 == current.code_sha256
        and previous.error_fingerprint == current.error_fingerprint
    ):
        return "unchanged"
    if previous.error_fingerprint and current.error_fingerprint:
        return "improved"
    if previous.code_sha256 != current.code_sha256:
        return "improved"
    return "unknown"


__all__ = ["ProgressKind", "ProgressSnapshot", "classify_progress"]
