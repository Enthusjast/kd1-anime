"""Dependency calculations over immutable scene snapshots, without workflow access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from kd1_anime.domain.dependencies import DependencyBinding, SceneDependency, digest
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.domain.technical import TechnicalSpec

if TYPE_CHECKING:
    from kd1_anime.run_store import RunManifest


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

    @staticmethod
    def affected_manifest(
        before: RunManifest, after: RunManifest, changed_scene_ids: Sequence[int]
    ) -> set[int]:
        """Compute invalidations from stored values, without mutating either run.

        A revision-only update is not an output change. Once a producer changes,
        the graph's conservative transitive rule also covers consumers whose
        future output contracts cannot yet be known. No job cancellation or file
        operation belongs in this calculation.
        """

        def node(scene):
            return {
                "revision": scene.revision,
                "code": scene.code_sha256,
                "contract": DependencyService.contract(scene.plan, scene.technical_spec),
                "teaching": DependencyService.teaching(scene.plan),
            }

        plans = [scene.plan for scene in before.scenes.values()]
        graph = before.dependency_graph
        bound = {binding.consumer_scene_id for binding in graph.bindings}
        fallback = DependencyService.fallback_edges(plans, bound, {})
        if after.dependency_graph.bindings != graph.bindings:
            fallback.extend(after.dependency_graph.bindings)
        affected = set()
        for scene_id in changed_scene_ids:
            previous = before.scenes.get(scene_id)
            proposed = after.scenes.get(scene_id)
            if previous is None or proposed is None:
                continue
            old_node = node(previous)
            new_node = node(proposed)
            if any(old_node[key] != new_node[key] for key in ("code", "contract", "teaching")):
                affected.update(graph.affected(scene_id, {scene_id: new_node}, fallback))

        def global_inputs(manifest):
            return digest(
                {
                    "lesson": manifest.lesson_spec.model_dump(mode="json"),
                    "teaching": manifest.teaching_graph.model_dump(mode="json"),
                    "bible": manifest.continuity_bible.model_dump(mode="json")
                    if manifest.continuity_bible is not None
                    else None,
                    "renderer": manifest.render_profile.digest(),
                    "rag": manifest.rag_profile.model_dump(mode="json"),
                }
            )

        if (
            before.lesson_spec != after.lesson_spec
            or before.teaching_graph != after.teaching_graph
            or before.continuity_bible != after.continuity_bible
            or before.render_profile != after.render_profile
            or before.rag_profile != after.rag_profile
        ):
            global_consumers = graph.changed_globals(global_inputs(after))
            affected.update(global_consumers)
            # Their future outputs are unknown, so every downstream dependency
            # must be invalidated, even if its producer's old digest still fits.
            frontier = set(global_consumers)
            edges = (*graph.bindings, *fallback)
            while frontier:
                source = frontier.pop()
                for edge in edges:
                    if edge.source_scene_id == source and edge.consumer_scene_id not in affected:
                        affected.add(edge.consumer_scene_id)
                        frontier.add(edge.consumer_scene_id)
        return affected
