from dataclasses import replace

import pytest
from test_pipeline_scene import SOURCE, inputs

from kd1_anime.domain.planning import VisualElementState
from kd1_anime.objects import ObjectStore
from kd1_anime.run_store import sha256_text
from kd1_anime.services.continuity_state import (
    build_continuity_publication,
    prepare_inherited_inputs,
)

EXPORT = """from manim import *
class Pause(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: shape
        shape = Circle()
        # KD1_CONTINUITY_EXPORT_END
        self.add(shape)
        self.wait(1)
"""


def scene_inputs(
    tmp_path, scene_id, *, inherited=False, removed=False, independent=False, reviewed=True
):
    value = inputs(tmp_path)
    element = VisualElementState(element_id="shape", variable_name="shape")
    plan = value.scene.plan.model_copy(
        update={
            "scene_id": scene_id,
            "dependencies_complete": independent,
            "new_elements": [element] if scene_id == 1 else [],
            "inherited_elements": [element] if inherited else [],
            "elements_to_remove": [element] if removed else [],
        }
    )
    text = EXPORT if scene_id == 1 or (inherited and not removed) else SOURCE
    if removed:
        text = SOURCE.replace(
            "self.wait(1)",
            "shape = Circle()\n        self.add(shape)\n        self.play(FadeOut(shape))",
        )
    scene = value.scene.model_copy(
        update={
            "plan": plan,
            "plan_ready": True,
            "reviewed": reviewed,
            "code_file": ObjectStore(tmp_path).source(scene_id, text),
            "code_sha256": sha256_text(text),
            "class_name": "Pause",
        }
    )
    return replace(value, scene=scene, plans=(plan,))


def test_publication_is_ordered_and_preserves_removed_history_without_shared_mutation(tmp_path):
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2, inherited=True, removed=True)
    before = first.element_manifest.model_dump()
    result = build_continuity_publication((first, second))
    assert result.element_manifest.entries == []
    assert result.state_ledger.boundaries[1].closing_element_ids == ["shape"]
    assert result.state_ledger.boundaries[2].opening_element_ids == ["shape"]
    assert result.state_ledger.boundaries[2].closing_element_ids == []
    assert not result.state_ledger.elements[0].active
    assert not result.state_ledger.elements[0].required_next
    assert first.element_manifest.model_dump() == before
    assert first.state_ledger.boundaries == {}
    assert first.scene.exported_elements == []
    assert result.dependency_bindings


def test_publication_requires_reviewed_source_and_order(tmp_path):
    first = scene_inputs(tmp_path, 1, reviewed=False)
    with pytest.raises(ValueError):
        build_continuity_publication((first,))
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2)
    with pytest.raises(ValueError):
        build_continuity_publication((second, first))


def test_inheritance_extracts_exact_exports_and_does_not_mutate_snapshots(tmp_path):
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2, inherited=True)
    prepared = prepare_inherited_inputs(second, (first,))
    assert "shape = Circle()" in prepared.scene.inherited_elements_code
    assert "class Pause" not in prepared.scene.inherited_elements_code
    assert prepared.scene is not second.scene
    assert not second.scene.inherited_elements_code
    assert not first.scene.exported_elements
    assert prepared.element_manifest.for_elements({"shape"})


def test_explicit_independence_has_no_implicit_code_context(tmp_path):
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2, independent=True)
    prepared = prepare_inherited_inputs(second, (first,))
    assert prepared.scene.inherited_elements_code == ""
    assert prepared.element_manifest.entries == []


def test_removed_tombstones_cannot_be_inherited_again(tmp_path):
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2, inherited=True, removed=True)
    third = scene_inputs(tmp_path, 3, inherited=True)
    with pytest.raises(ValueError):
        prepare_inherited_inputs(third, (first, second))


def test_tampered_predecessor_is_never_an_inheritance_source(tmp_path):
    first = scene_inputs(tmp_path, 1)
    second = scene_inputs(tmp_path, 2, inherited=True)
    (tmp_path / first.scene.code_file).write_text("tampered")
    with pytest.raises(ValueError):
        prepare_inherited_inputs(second, (first,))


def test_inheritance_reads_legacy_exports_without_fabricating_declarations(tmp_path):
    first = scene_inputs(tmp_path, 1)
    first = replace(
        first,
        scene=first.scene.model_copy(
            update={
                "plan": first.scene.plan.model_copy(update={"new_elements": []}),
            }
        ),
    )
    second = scene_inputs(tmp_path, 2, inherited=True)
    prepared = prepare_inherited_inputs(second, (first.scene,))
    assert "shape = Circle()" in prepared.scene.inherited_elements_code
    assert first.scene.plan.new_elements == []
