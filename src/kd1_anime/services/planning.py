"""计划阶段的纯辅助逻辑。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kd1_anime.agents.continuity import normalize_scene_plan_contract
from kd1_anime.agents.plan_compiler import normalize_scene_timeline_contract
from kd1_anime.domain.planning import ContinuityBible, LessonSpec, SceneOutline, ScenePlan
from kd1_anime.domain.technical import TechnicalHandoff, TechnicalSpec
from kd1_anime.models import snapshot_value, validated_copy


@dataclass(frozen=True, slots=True)
class PlannedScene:
    plan: ScenePlan


class PlanningService:
    """集中处理不依赖 FSM 的计划指纹和时长计算。"""

    @staticmethod
    def detail(
        agent,
        outline: SceneOutline,
        outlines: Sequence[SceneOutline],
        prompt: str,
        options: Mapping[str, Any],
    ) -> PlannedScene:
        plan = agent.plan_detail(
            validated_copy(outline, deep=True),
            [validated_copy(item, deep=True) for item in outlines],
            prompt,
            **snapshot_value(dict(options)),
        )
        if not isinstance(plan, ScenePlan) or plan.scene_id != outline.scene_id:
            raise ValueError("Planner returned a plan for a different scene")
        return PlannedScene(validated_copy(plan, deep=True))

    @staticmethod
    def normalize(
        plan: ScenePlan,
        bible: ContinuityBible,
        lesson_spec: LessonSpec,
        *,
        previous_plan: ScenePlan | None = None,
        has_next_scene: bool,
    ) -> tuple[ScenePlan, tuple[str, ...]]:
        """Restore declared contracts, never invent or discard mathematical facts."""
        plan = validated_copy(plan, deep=True)
        plan, continuity_repairs = normalize_scene_plan_contract(
            plan,
            validated_copy(bible, deep=True),
            previous_plan=validated_copy(previous_plan, deep=True)
            if previous_plan is not None
            else None,
            has_next_scene=has_next_scene,
        )
        repairs = list(continuity_repairs)
        approved = {claim.claim_id: claim for claim in lesson_spec.claims}
        present = {claim.claim_id for claim in plan.math_claims}
        missing = [
            claim_id
            for claim_id in plan.claim_ids
            if claim_id in approved and claim_id not in present
        ]
        claims = [
            *plan.math_claims,
            *(validated_copy(approved[item], deep=True) for item in missing),
        ]
        if missing:
            repairs.append("Restored LessonSpec claims: " + ", ".join(missing))
        # Outline-assigned facts may be restored from the immutable LessonSpec;
        # unassigned or conflicting facts remain visible to the compiler.
        events = list(plan.timeline)
        bound = {claim_id for event in events for claim_id in event.math_claim_ids}
        for claim in claims:
            if claim.claim_id not in plan.claim_ids or claim.claim_id in bound or not events:
                continue
            text = " ".join((claim.statement, claim.expression_before, claim.expression_after))
            terms = set(
                re.findall(r"[a-z][a-z0-9_]{1,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", text.lower())
            )

            def score(index: int, terms=terms) -> tuple[int, float, float]:
                event = events[index]
                action = event.action.lower()
                return (
                    sum(3 for term in terms if term in action),
                    event.end_seconds - event.start_seconds,
                    -event.start_seconds,
                )

            index = max(range(len(events)), key=score)
            event = events[index]
            events[index] = event.model_copy(
                update={"math_claim_ids": [*event.math_claim_ids, claim.claim_id]}
            )
            bound.add(claim.claim_id)
            repairs.append("Bound LessonSpec claim to timeline: " + claim.claim_id)
        if missing or events != plan.timeline:
            plan = plan.model_copy(update={"math_claims": claims, "timeline": events})
        plan, timeline_repairs = normalize_scene_timeline_contract(plan)
        return validated_copy(plan, deep=True), (*repairs, *timeline_repairs)

    @staticmethod
    def normalize_plans(
        plans: Sequence[ScenePlan],
        bible: ContinuityBible,
        lesson_spec: LessonSpec,
    ) -> tuple[tuple[ScenePlan, ...], dict[int, tuple[str, ...]]]:
        """Normalize complete ordered plans without mutating worker-owned snapshots."""
        ordered = sorted(plans, key=lambda item: item.scene_id)
        normalized: list[ScenePlan] = []
        repairs: dict[int, tuple[str, ...]] = {}
        for index, plan in enumerate(ordered):
            current, messages = PlanningService.normalize(
                plan,
                bible,
                lesson_spec,
                previous_plan=normalized[-1] if normalized else None,
                has_next_scene=index + 1 < len(ordered),
            )
            if index + 1 < len(ordered):
                consumed = {item.element_id for item in ordered[index + 1].inherited_elements}
                new_ids = {item.element_id for item in current.new_elements}
                dangling = {
                    item.element_id
                    for item in current.handoff
                    if item.action in {"inherit", "keep", "create"}
                    and item.element_id in new_ids
                    and item.element_id not in consumed
                }
                if dangling:
                    current = current.model_copy(
                        update={
                            "new_elements": [
                                item.model_copy(update={"required": False})
                                if item.element_id in dangling
                                else item
                                for item in current.new_elements
                            ],
                            "handoff": [
                                item for item in current.handoff if item.element_id not in dangling
                            ],
                        }
                    )
                    messages = (
                        *messages,
                        "Made unconsumed new handoff elements scene-local: "
                        + ", ".join(sorted(dangling)),
                    )
            normalized.append(validated_copy(current, deep=True))
            if messages:
                repairs[current.scene_id] = messages
        return tuple(normalized), repairs

    @staticmethod
    def technical_handoff_errors(
        plan: ScenePlan,
        spec: TechnicalSpec,
        previous: TechnicalHandoff | None,
    ) -> tuple[str, ...]:
        """Check inherited identities against the accepted prior technical boundary."""
        inherited_ids = {item.element_id for item in plan.inherited_elements}
        if previous is None:
            return (
                ("Inherited objects require previous_technical_handoff",) if inherited_ids else ()
            )
        errors: list[str] = []
        if previous.source_scene_id != plan.scene_id - 1:
            errors.append("previous_technical_handoff.source_scene_id is not the preceding scene")
        if spec.handoff_in is not None and spec.handoff_in != previous:
            errors.append("TechnicalSpec.handoff_in differs from previous_technical_handoff")
        upstream = {item.element_id: item for item in previous.elements}
        objects = {item.element_id: item for item in spec.objects}
        for element_id in sorted(inherited_ids):
            before, after = upstream.get(element_id), objects.get(element_id)
            if before is None:
                errors.append(f"Inherited object {element_id} has no previous technical export")
                continue
            if after is None:
                errors.append(
                    f"Inherited object {element_id} is missing from TechnicalSpec.objects"
                )
                continue
            if before.variable_name != after.variable_name:
                errors.append(
                    f"Inherited object {element_id} variable_name differs from the prior boundary"
                )
            if (
                before.constructor.split("(", 1)[0].strip()
                != after.constructor.split("(", 1)[0].strip()
            ):
                errors.append(
                    f"Inherited object {element_id} constructor differs from the prior boundary"
                )
            if (
                before.final_state.strip()
                and before.final_state.strip() != after.initial_state.strip()
            ):
                errors.append(
                    f"Inherited object {element_id} initial_state differs from the prior final_state"
                )
        return tuple(errors)

    @staticmethod
    def cycle_signature(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def expected_duration(
        durations: Sequence[float],
        transition_duration: float,
    ) -> float:
        values = [float(duration) for duration in durations]
        if not values:
            return 0.0
        transition = min(
            max(0.0, float(transition_duration)),
            max(0.0, min(values) / 2) if len(values) > 1 else 0.0,
        )
        return max(0.0, sum(values) - transition * max(0, len(values) - 1))


__all__ = ["PlanningService"]
