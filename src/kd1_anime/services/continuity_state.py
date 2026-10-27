"""Pure inheritance snapshots and ordered publication of reviewed scene boundaries.

Workers may prepare private inputs here; only the coordinator publishes the returned
ledgers. Source extraction never edits predecessor snapshots or any shared manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from kd1_anime.agent_runtime.contracts import RuntimeRecord
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.domain.dependencies import DependencyBinding, DependencyGraph
from kd1_anime.domain.diagnostics import Diagnostic
from kd1_anime.domain.planning import ElementManifest
from kd1_anime.domain.state_ledger import LedgerElement, StateLedger, validate_boundary_handoff
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import StoredSceneState, restore_run_path, sha256_text
from kd1_anime.services.dependencies import DependencyService


class ContinuityPublication(RuntimeRecord):
    element_manifest: ElementManifest
    state_ledger: StateLedger
    diagnostics: tuple[Diagnostic, ...] = ()
    dependency_bindings: tuple[DependencyBinding, ...] = ()


def _accepted_exports(inputs: SceneInputs):
    scene = inputs.scene
    if not scene.reviewed:
        raise ValueError(f"Scene {scene.plan.scene_id} source has not passed review")
    path = restore_run_path(inputs.root, scene.code_file)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != scene.code_sha256:
        raise ValueError("Continuity source identity mismatch")
    with inputs.config.activate():
        inspection = CandidateAcceptor().inspect_candidate(
            path.read_text(encoding="utf-8"),
            scene.plan,
            technical_spec=scene.technical_spec,
            renderer=inputs.render_profile.renderer,
            manim_version=inputs.render_profile.manim_version,
            expected_class_name=scene.class_name or None,
            allow_repair=False,
        )
    if not inspection.accepted:
        raise ValueError("Continuity source inspection failed: " + inspection.feedback)
    elements = [
        item.model_copy(
            update={
                "source_scene_id": scene.plan.scene_id,
                "source_code_sha256": scene.code_sha256,
            }
        )
        for item in inspection.exported_elements
    ]
    return inspection.exported_elements_code, elements


def prepare_inherited_inputs(
    inputs: SceneInputs,
    prior_snapshots: Sequence[SceneInputs | StoredSceneState],
) -> SceneInputs:
    """Return a private scene snapshot containing only its declared source context.

    Incomplete dependency declarations retain the historical adjacent-scene code
    handoff. Explicitly independent scenes never acquire incidental sibling code.
    Removed elements are resolved before selection, so old exports cannot resurrect
    a tombstone even when a prior immutable object still exists on disk.
    """
    scene = inputs.scene
    plan = scene.plan
    prior = sorted(
        (
            item if isinstance(item, SceneInputs) else replace(inputs, scene=item)
            for item in prior_snapshots
            if (item.scene.plan.scene_id if isinstance(item, SceneInputs) else item.plan.scene_id)
            < plan.scene_id
        ),
        key=lambda item: item.scene.plan.scene_id,
    )
    ids = [item.scene.plan.scene_id for item in prior]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate predecessor scene identity")
    required = {item.element_id for item in plan.inherited_elements}
    declared_code = {
        item.source_scene_id for item in plan.dependencies if item.kind == "code_context"
    }
    legacy = not plan.dependencies_complete
    if not required and not declared_code and not legacy:
        snapshot = scene.model_copy(update={"inherited_elements_code": ""}, deep=True)
        return replace(
            inputs, scene=snapshot, element_manifest=ElementManifest(), state_ledger=StateLedger()
        )
    if not prior:
        if required or declared_code or (legacy and plan.scene_id > 1):
            raise ValueError(f"Scene {plan.scene_id} has no verified inheritance predecessor")
        return replace(
            inputs, scene=scene.model_copy(update={"inherited_elements_code": ""}, deep=True)
        )

    # Only inspect relevant providers. Still process removal declarations from all
    # prior plans: a later tombstone dominates an older provider's immutable export.
    prior_by_id = {item.scene.plan.scene_id: item for item in prior}
    providers = {
        entry.element_id: prior_by_id[entry.source_scene_id]
        for entry in inputs.element_manifest.entries
        if entry.source_scene_id in prior_by_id
    }
    removed = set()
    for item in prior:
        previous = item.scene.plan
        removed_here = {element.element_id for element in previous.elements_to_remove}
        removed.update(removed_here)
        for element in (
            *previous.new_elements,
            *previous.inherited_elements,
            *item.scene.exported_elements,
        ):
            if element.element_id not in removed_here:
                providers[element.element_id] = item
                removed.discard(element.element_id)
    if required & removed:
        raise ValueError("继承元素已被先行场景移除: " + ", ".join(sorted(required & removed)))
    selected_sources = {
        providers[element_id].scene.plan.scene_id
        for element_id in required
        if element_id in providers
    }
    if required - providers.keys():
        # Historical plans may have only a source export marker, without a
        # structured declaration or cached extraction. Inspect the actual adjacent
        # reviewed source rather than inventing a declaration or trusting text.
        selected_sources.add(prior[-1].scene.plan.scene_id)
    selected_sources.update(declared_code)
    if legacy and not required:
        selected_sources.add(prior[-1].scene.plan.scene_id)
    missing_sources = selected_sources - set(ids)
    if missing_sources:
        raise ValueError(
            "Missing declared code-context predecessor: "
            + ", ".join(map(str, sorted(missing_sources)))
        )
    manifest = ElementManifest()
    exports = {}
    for item in prior:
        if item.scene.plan.scene_id not in selected_sources:
            continue
        exported_code, elements = _accepted_exports(item)
        exports[item.scene.plan.scene_id] = exported_code
        manifest = manifest.update_scene(item.scene.plan, elements)
    selected = manifest.for_elements(required)
    missing = required - {entry.element_id for entry in selected}
    if missing:
        raise ValueError("所需继承元素没有可验证的导出记录: " + ", ".join(sorted(missing)))
    mode = inputs.config.resolve().CONTINUITY_CONTEXT_MODE
    if required:
        inherited_code = (
            "\n\n".join(exports[source] for source in sorted(exports))
            if mode == "full"
            else "\n\n".join(entry.source_code for entry in selected)
        )
        # Explicit code_context dependencies are semantic inputs even in minimal
        # mode, not incidental prior-scene data.
        additional = [
            exports[source]
            for source in sorted(declared_code - {entry.source_scene_id for entry in selected})
            if exports[source]
        ]
        if additional and mode != "full":
            inherited_code = "\n\n".join((inherited_code, *additional))
    else:
        inherited_code = (
            ""
            if mode == "stateless" and not declared_code
            else "\n\n".join(exports[source] for source in sorted(exports) if exports[source])
        )
    if required and not inherited_code.strip():
        raise ValueError("继承元素导出记录为空，禁止继续编码")
    snapshot = scene.model_copy(update={"inherited_elements_code": inherited_code}, deep=True)
    # Retain no unrelated global ledger or manifest in a worker's semantic inputs.
    ledger_ids = required | {entry.element_id for entry in manifest.entries}
    ledger = StateLedger(
        elements=[
            item.model_copy(deep=True)
            for item in inputs.state_ledger.elements
            if item.element_id in ledger_ids and item.active
        ]
    )
    return replace(inputs, scene=snapshot, element_manifest=manifest, state_ledger=ledger)


def build_continuity_publication(ordered_inputs: Sequence[SceneInputs]) -> ContinuityPublication:
    """Rebuild shared ledgers from reviewed immutable sources, in scene order."""
    ids = [item.scene.plan.scene_id for item in ordered_inputs]
    if ids != sorted(set(ids)):
        raise ValueError("Continuity publication requires unique scene order")
    manifest = ElementManifest()
    ledger = StateLedger()
    graph = DependencyGraph()
    nodes = {}
    previous_boundary = None
    for inputs in ordered_inputs:
        scene = inputs.scene
        plan = scene.plan
        exported_code, elements = _accepted_exports(inputs)
        manifest = manifest.update_scene(plan, elements)
        declarations = {
            item.element_id: item for item in (*plan.inherited_elements, *plan.new_elements)
        }
        new_elements = []
        for element in elements:
            declaration = declarations.get(element.element_id)
            new_elements.append(
                LedgerElement(
                    element_id=element.element_id,
                    variable_name=element.variable_name,
                    semantic_state=declaration.semantic_state if declaration else "",
                    mathematical_state="；".join(plan.closing_state),
                    color_key=declaration.color_key if declaration else "",
                    anchor=declaration.anchor if declaration else "",
                    active=True,
                    required_next=bool(declaration and declaration.required),
                    source_scene_id=plan.scene_id,
                    source_code_sha256=scene.code_sha256,
                    export_code_sha256=sha256_text(element.code),
                )
            )
        removed = {item.element_id for item in plan.elements_to_remove}
        ledger = ledger.model_copy(
            update={
                "elements": [
                    item.model_copy(update={"active": False, "required_next": False})
                    if item.element_id in removed
                    else item
                    for item in ledger.elements
                ]
            }
        )
        ledger = ledger.update_scene(
            scene_id=plan.scene_id,
            elements=new_elements,
            opening_element_ids=[item.element_id for item in plan.inherited_elements],
            closing_element_ids=[item.element_id for item in elements],
            opening_math_state="；".join(plan.opening_state),
            closing_math_state="；".join(plan.closing_state),
            removed_element_ids=sorted(removed),
            transition_in=plan.transition_in,
            transition_out=plan.transition_out,
            visual_state_digest=sha256_text(plan.global_visual_state.model_dump_json()),
            exported_code_sha256=sha256_text(exported_code) if exported_code else "",
            artifact_video_sha256=scene.artifact.video_sha256 if scene.artifact else "",
        )
        boundary = ledger.boundaries[plan.scene_id]
        errors = validate_boundary_handoff(previous_boundary, boundary)
        if errors:
            raise ValueError("场景边界状态合同无效: " + "；".join(errors))
        dependencies = DependencyService.dependencies(
            plan,
            inputs.plans,
            inherited_code=scene.inherited_elements_code,
            handoff_source=previous_boundary.scene_id if previous_boundary is not None else None,
        )
        graph.bind(plan.scene_id, dependencies, nodes)
        nodes[plan.scene_id] = {
            "revision": scene.revision,
            "code": scene.code_sha256,
            "contract": DependencyService.contract(plan, scene.technical_spec),
            "teaching": DependencyService.teaching(plan),
        }
        previous_boundary = boundary
    return ContinuityPublication(
        element_manifest=manifest,
        state_ledger=ledger,
        dependency_bindings=tuple(graph.bindings),
    )
