"""Dependency calculations over immutable scene snapshots, without workflow access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kd1_anime.domain.dependencies import DependencyBinding, SceneDependency, digest
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.domain.technical import TechnicalSpec


class DependencyService:
    @staticmethod
    def contract(plan: ScenePlan, technical: TechnicalSpec | None) -> str:
        removed = {item.element_id for item in plan.elements_to_remove}
        elements = [
            item.model_dump(mode="json")
            for item in (*plan.inherited_elements, *plan.new_elements)
            if item.required and item.element_id not in removed
        ]
        technical_exports = []
        if technical is not None:
            exported = set(technical.export_element_ids)
            technical_exports = [
                item.model_dump(mode="json")
                for item in technical.objects
                if item.element_id in exported or item.exported
            ]
        return digest({"elements": elements, "technical": technical_exports})

    @staticmethod
    def teaching(plan: ScenePlan) -> str:
        return digest(
            {
                "claims": [item.model_dump(mode="json") for item in plan.math_claims],
                "claim_ids": plan.claim_ids,
                "concept": plan.math_concept,
                "purpose": plan.purpose,
                "computation": plan.computation,
            }
        )

    @staticmethod
    def dependencies(
        plan: ScenePlan,
        plans: Sequence[ScenePlan],
        *,
        inherited_code: str = "",
        handoff_source: int | None = None,
    ) -> list[SceneDependency]:
        result = list(plan.dependencies)
        producers = {
            element.element_id: item.scene_id
            for item in plans
            if item.scene_id < plan.scene_id
            for element in item.new_elements
        }
        for element in plan.inherited_elements:
            source = producers.get(element.element_id, handoff_source)
            if source is not None:
                result.append(
                    SceneDependency(
                        source_scene_id=source, kind="handoff", element_ids=[element.element_id]
                    )
                )
        previous = [item.scene_id for item in plans if item.scene_id < plan.scene_id]
        if inherited_code and previous:
            result.append(
                SceneDependency(
                    source_scene_id=handoff_source or max(previous), kind="code_context"
                )
            )
        if not plan.dependencies_complete and previous:
            # A missing declaration cannot be interpreted as independence.
            result.append(
                SceneDependency(
                    source_scene_id=max(previous),
                    kind="handoff"
                    if handoff_source is not None and not inherited_code
                    else "code_context",
                )
            )
        unique = {(item.source_scene_id, item.kind): item for item in result}
        return list(unique.values())

    @staticmethod
    def fallback_edges(
        plans: Sequence[ScenePlan], bound_consumers: set[int], nodes: Mapping[int, dict]
    ) -> list[DependencyBinding]:
        result = []
        for plan in plans:
            if plan.scene_id in bound_consumers:
                continue
            for dependency in DependencyService.dependencies(plan, plans):
                result.append(
                    DependencyBinding(
                        source_scene_id=dependency.source_scene_id,
                        consumer_scene_id=plan.scene_id,
                        kind=dependency.kind,
                        source_sha256="0" * 64,
                    )
                )
        return result
