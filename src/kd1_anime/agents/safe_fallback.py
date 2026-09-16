"""面向高风险几何分镜的保守降级策略。

复杂的切割、旋转和面积拼接很难仅凭自然语言保证几何正确。这个模块不
尝试替 Planner 发明新的几何证明，而是在确定性风险和有限审查预算耗尽
时，把分镜收敛到“基础图形 + 面积/等式 + 结论”的可执行教学方案。
"""

from __future__ import annotations

import re

from kd1_anime.agents.planner import (
    ContinuityBible,
    ScenePlan,
    TimelineEvent,
    VisualElementState,
)

_GEOMETRY_TERMS = (
    "切割",
    "碎片",
    "拼接",
    "拼成",
    "无缝",
    "重新组合",
    "移动到目标",
    "piece_",
    "fragment",
    "reassembled",
    "cut_line",
)
_FAILURE_TERMS = (
    "无法",
    "错误",
    "不正确",
    "不合理",
    "未验证",
    "不一致",
    "重叠",
    "超出",
    "越界",
    "安全范围",
    "面积",
    "geometry",
    "geometric",
)
_TRANSIENT_ELEMENT_TERMS = (
    "piece",
    "fragment",
    "碎片",
    "cut_line",
    "切割线",
    "辅助线",
    "highlight",
    "高亮框",
)


def is_high_confidence_geometry_conflict(plan: ScenePlan, feedback: str) -> bool:
    """判断是否应从复杂几何降级，而不是继续重复同一个方案。

    仅有“几何”二字不足以触发降级；必须同时出现几何动作和明确的
    失败/不可验证信号，或分镜本身已经声明了碎片/重组对象。
    """

    plan_text = "\n".join(
        (
            plan.visual_design,
            *plan.visual_flow,
            plan.computation,
            " ".join(
                f"{item.geometry_id} {item.shape} {item.target_description}"
                for item in plan.geometry_specs
            ),
            " ".join(item.element_id for item in plan.new_elements),
        )
    ).lower()
    feedback_text = (feedback or "").lower()
    has_geometry = any(term.lower() in plan_text for term in _GEOMETRY_TERMS)
    feedback_marks_geometry = "[geometry]" in feedback_text or any(
        term in feedback_text for term in ("几何", "顶点", "正方形", "三角形")
    )
    if feedback_marks_geometry and (
        any(term in plan_text for term in ("正方形", "三角形", "坐标", "顶点", "面积"))
        or bool(plan.geometry_specs)
    ):
        has_geometry = True
    has_structured_risk = any(
        any(term.lower() in item.element_id.lower() for term in _GEOMETRY_TERMS)
        for item in plan.new_elements
    )
    has_failure = any(term.lower() in feedback_text for term in _FAILURE_TERMS)
    return (has_geometry and has_failure) or has_structured_risk


def _is_transient(item: VisualElementState) -> bool:
    text = f"{item.element_id} {item.variable_name} {item.role}".lower()
    return any(term.lower() in text for term in _TRANSIENT_ELEMENT_TERMS)


def build_safe_fallback_plan(
    plan: ScenePlan,
    bible: ContinuityBible,
    *,
    reason: str = "复杂几何方案无法可靠验证",
) -> ScenePlan:
    """把高风险分镜改写为不声称错误几何结论的保守教学分镜。"""

    allowed_colors = set(bible.global_visual_state.colors)
    fallback_color = (
        "primary" if "primary" in allowed_colors else next(iter(sorted(allowed_colors)), "")
    )

    def normalize_element(item: VisualElementState) -> VisualElementState:
        updates: dict[str, object] = {}
        if item.color_key and item.color_key not in allowed_colors:
            updates["color_key"] = fallback_color
        if item.element_id.lower().startswith(("reassembled", "assembled")):
            updates["role"] = "面积关系示意区域"
            updates["semantic_state"] = (
                "面积关系示意区域；仅表达等式关系，不表示未经计算的无缝碎片拼接"
            )
            if item.color_key not in allowed_colors:
                updates["color_key"] = fallback_color
        if _is_transient(item):
            updates["required"] = False
            updates["reason"] = "保守方案不交接临时几何对象"
        return item.model_copy(update=updates) if updates else item

    inherited = [normalize_element(item) for item in plan.inherited_elements]
    removals = [normalize_element(item) for item in plan.elements_to_remove]
    new_elements = [normalize_element(item) for item in plan.new_elements]
    inherited_ids = {item.element_id for item in inherited}
    removals = [item for item in removals if item.element_id in inherited_ids]
    removal_ids = {item.element_id for item in removals}
    new_elements = [
        item
        for item in new_elements
        if item.element_id not in inherited_ids and item.element_id not in removal_ids
    ]

    retained = [
        item
        for item in [*inherited, *new_elements]
        if item.element_id not in removal_ids and not _is_transient(item)
    ]
    retained_labels = (
        "、".join(item.role or item.element_id for item in retained[:8]) or "已确认的视觉对象"
    )
    fallback_flow = [
        f"直接接管并稳定展示：{retained_labels}。",
        "不执行未经坐标、面积和目标位置验证的切割、旋转或无缝拼接。",
        "使用颜色编码、面积标签或等式变换表达核心数学关系。",
        "保留公式和结论到场景结束，交给下一场景继续接管。",
    ]
    # 保守方案不是把原计划原样复制后换几句说明：原计划的长时间静止、
    # 越界顶点和错误几何规格仍会被 PlanCompiler 检出。将时长限制在一个
    # 可教学且可落地的范围，并重建一条只引用保留元素的最小时间线。
    duration = min(max(plan.duration_seconds, 0.1), 75.0)
    checkpoints = [0.0, duration * 0.2, duration * 0.53, duration * 0.8, duration]
    fallback_moments = [
        f"{checkpoints[0]:g}-{checkpoints[1]:g}s — 接管已确认对象并稳定构图 — 停留 1s",
        f"{checkpoints[1]:g}-{checkpoints[2]:g}s — 展示面积标签或等式关系 — 停留 1s",
        f"{checkpoints[2]:g}-{checkpoints[3]:g}s — 高亮核心公式和结论 — 停留 1s",
        f"{checkpoints[3]:g}-{checkpoints[4]:g}s — 保持最终状态并完成交接 — 停留 2s",
    ]
    declared_element_ids = [item.element_id for item in retained]
    # 只有计划已有的断言才能继续绑定到画面证据；不凭空制造新的数学
    # 断言。若上游规范器已把 claim_ids 与 math_claims 对齐，这里会自然
    # 保留完整的教学合同；若旧计划没有结构化断言，则保持空列表。
    declared_claim_ids = {claim.claim_id for claim in plan.math_claims}
    timeline_claim_ids = [claim_id for claim_id in plan.claim_ids if claim_id in declared_claim_ids]
    fallback_timeline = [
        TimelineEvent(
            event_id="fallback_intro",
            start_seconds=checkpoints[0],
            end_seconds=checkpoints[1],
            action="接管已确认对象并稳定构图",
            element_ids=declared_element_ids,
            math_claim_ids=timeline_claim_ids,
        ),
        TimelineEvent(
            event_id="fallback_relation",
            start_seconds=checkpoints[1],
            end_seconds=checkpoints[2],
            action="使用面积标签或等式变换表达核心数学关系",
            element_ids=declared_element_ids,
            math_claim_ids=timeline_claim_ids,
        ),
        TimelineEvent(
            event_id="fallback_conclusion",
            start_seconds=checkpoints[2],
            end_seconds=checkpoints[3],
            action="高亮核心公式和结论",
            element_ids=declared_element_ids,
            math_claim_ids=timeline_claim_ids,
        ),
        TimelineEvent(
            event_id="fallback_hold",
            start_seconds=checkpoints[3],
            end_seconds=checkpoints[4],
            action="保持最终状态并完成场景交接",
            element_ids=declared_element_ids,
            math_claim_ids=timeline_claim_ids,
            pause_seconds=2,
        ),
    ]
    retained_state = retained_labels
    retained_elements_state = [
        f"{item.role or item.element_id}（element_id={item.element_id}）" for item in retained
    ] or [retained_labels]
    return plan.model_copy(
        update={
            "purpose": f"{plan.purpose}（已切换为保守教学表达）"[:5_000],
            "visual_design": (
                "沿用全片连续性圣经的背景、颜色、字体和布局。"
                "只展示已经确认的基础图形、面积标签和等式，不伪造未验证的几何拼接。"
            ),
            "camera_movement": "固定中景；通过高亮面积标签和公式转移焦点，不推拉或切换机位。",
            "visual_flow": fallback_flow,
            "key_moments": fallback_moments,
            "computation": (
                f"保守实现约束：{reason}。"
                "只使用分镜中已经给出的可验证公式、面积或变量关系；"
                "不添加未经计算的碎片顶点、旋转角度或目标坐标。"
            ),
            # 复杂几何规格是触发降级的根因；继续保留它们会让确定性
            # 编译器再次验证原错误，从而出现“已降级仍无法通过”的死循环。
            "geometry_specs": [],
            "persistent_elements": retained_elements_state,
            "opening_state": ["接管已确认的视觉对象：" + retained_labels],
            "closing_state": ["保留已确认的视觉对象和核心数学关系：" + retained_state],
            "transition_in": (
                "直接接管上一场景已确认的对象和数学状态；不清空画面，不重新制造未经验证的几何起点。"
            ),
            "transition_out": (
                f"保留已确认对象和核心公式（{retained_state}），下一场景直接接管这些状态。"
            ),
            "continuity_references": list(plan.continuity_references)
            or ["严格使用全片连续性圣经的颜色、字体、字号和布局"],
            "global_visual_state": bible.global_visual_state.model_copy(deep=True),
            "inherited_elements": inherited,
            "elements_to_remove": removals,
            "new_elements": new_elements,
            "timeline": fallback_timeline,
            "duration_seconds": duration,
        }
    )


def fallback_reason_summary(feedback: str, *, limit: int = 500) -> str:
    """生成可写入 manifest/UI 的短原因，不保留大段 LLM 原文。"""

    compact = re.sub(r"\s+", " ", feedback or "").strip()
    return compact[:limit] or "复杂几何方案未能通过有限审查"
