"""Pure acceptance, candidate and repair-routing rules."""

from __future__ import annotations

from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.review import ReviewFinding, ReviewResult


def accepts_review(result: ReviewResult, *, relaxed: bool, allow_soft_pass: bool) -> bool:
    return result.is_valid or (
        relaxed and allow_soft_pass and not result.findings and not result.fixes
    )


def retry_allowed(used: int, limit: int) -> bool:
    return 0 <= used < limit


def review_finding_requires_plan_repair(finding: ReviewFinding) -> bool:
    """区分“计划本身错误”和“代码实现了错误数学/连续性动作”。

    Reviewer 的 ``math``/``continuity`` 分类同时覆盖两种问题：例如
    “分镜要求错误公式”应回到 Planner，而“代码用 apply_function 实现
    线性剪切”应留在 Coder 修复。后者具有当前代码的行号、证据和 API
    级 repair；若仅按 category 路由，会把代码错误误送回计划层，造成
    重规划并丢失本来可以局部修复的正确分镜。
    """

    text = " ".join((finding.why, finding.repair)).lower()
    plan_markers = (
        "sceneplan",
        "lessonspec",
        "teachinggraph",
        "教学合同",
        "数学合同",
        "计划层",
        "计划本身",
        "当前计划",
        "分镜本身",
        "重新规划",
        "重规划",
        "无法通过代码修复",
        "claim_id 分配",
    )
    return any(marker.lower() in text for marker in plan_markers)


def candidate_rank(candidate: CodeCandidate) -> tuple[int, float]:
    verification_rank = {"validated": 1, "smoke": 2, "rendered": 3}.get(
        candidate.verification,
        0,
    )
    return verification_rank, candidate.visual_score or 0.0


def visual_repair_target(result: object) -> str:
    """根据视觉问题类别决定回到计划层还是代码层。"""

    issues = list(getattr(result, "issues", []) or [])
    for issue in issues:
        target = getattr(issue, "repair_target", "unknown")
        if target in {"planner", "continuity"}:
            return target
    categories = {getattr(issue, "category", "other") for issue in issues}
    if "mathematics" in categories or "relevance" in categories:
        return "planner"
    if "consistency" in categories:
        return "continuity"
    if categories & {"readability", "layout", "clipping", "overlap", "contrast"}:
        return "coder"
    # 没有 issue 但整体维度低时，保留旧版“质量低→代码优化”的行为；
    # 显式标记为 other/unknown 的问题则不应盲目修改代码。
    return "coder" if not issues else "unknown"
