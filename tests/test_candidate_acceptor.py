from pathlib import Path

import pytest

from kd1_anime.agents.planner import ScenePlan, VisualElementState
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec
from kd1_anime.candidate_acceptor import CandidateAcceptor, CandidateRejected


def make_plan() -> ScenePlan:
    return ScenePlan(
        scene_id=1,
        title="测试",
        duration_seconds=1,
        purpose="验证候选入口",
        math_concept="等待",
        visual_design="简洁",
        camera_movement="固定",
        visual_flow=["等待"],
        key_moments=["等待"],
        computation="无",
    )


CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.wait()\n"


def test_candidate_acceptor_returns_hash_and_can_write_atomically(tmp_path: Path):
    destination = tmp_path / "scene.py"
    accepted = CandidateAcceptor().accept_candidate(CODE, make_plan(), destination=destination)

    assert accepted.class_name == "Demo"
    assert accepted.code_sha256
    assert destination.read_text(encoding="utf-8") == CODE
    assert accepted.exported_elements == ()


def test_candidate_acceptor_rejects_unsafe_code():
    with pytest.raises(CandidateRejected, match="AST/安全校验"):
        CandidateAcceptor().accept_candidate("import os\nos.system('rm -rf /')", make_plan())


def test_candidate_acceptor_records_and_returns_deterministic_lifecycle_repairs():
    technical = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="MathTex",
                exported=True,
            )
        ],
        animations=[
            {
                "event_id": "show_formula",
                "start_seconds": 0,
                "end_seconds": 1,
                "semantic_action": "introduce",
                "target_element_ids": ["formula"],
                "create_element_ids": ["formula"],
            }
        ],
        export_element_ids=["formula"],
    )
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x", tex_template=tex_template)
        # KD1_CONTINUITY_EXPORT_END
        self.play(FadeIn(formula))
"""

    plan = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="formula", variable_name="formula", required=True)
            ]
        }
    )
    accepted = CandidateAcceptor().accept_candidate(code, plan, technical_spec=technical)

    assert "# KD1_ANIMATION_EVENT: show_formula" in accepted.code


def vector_candidate(body: str, *, exported: bool = True) -> str:
    export = (
        "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
        "        # element_id: grid\n"
        "        grid = Vector(RIGHT)\n"
        "        # KD1_CONTINUITY_EXPORT_END\n"
        if exported
        else "        grid = Vector(RIGHT)\n"
    )
    return "from manim import *\nclass Demo(Scene):\n    def construct(self):\n" + export + body


def vector_contract(*, active=False, removed=False, event=None):
    plan = make_plan().model_copy(
        update={
            "new_elements": []
            if removed
            else [VisualElementState(element_id="grid", variable_name="grid", required=True)]
        }
    )
    animations = []
    if event:
        animations = [
            {
                "event_id": "grid_event",
                "start_seconds": 0,
                "end_seconds": 1,
                "semantic_action": event,
                "source_element_ids": ["grid"] if event == "update" else [],
                "target_element_ids": ["grid"] if event == "introduce" else [],
                "create_element_ids": ["grid"] if event == "introduce" else [],
            }
        ]
    technical = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="grid",
                variable_name="grid",
                constructor="Vector",
                initially_active=active,
                exported=not removed,
            )
        ],
        export_element_ids=[] if removed else ["grid"],
        removed_element_ids=["grid"] if removed else [],
        animations=animations,
    )
    return plan, technical


@pytest.mark.parametrize(
    ("body", "active", "event", "expected", "forbidden"),
    [
        (
            "        self.play(GrowArrow(grid))\n",
            False,
            None,
            "Create(grid)",
            "GrowArrow(grid)",
        ),
        (
            "        grid_initial = Vector(RIGHT)\n"
            "        self.play(FadeIn(grid_initial))\n"
            "        grid = grid_initial\n"
            "        self.play(grid.animate.scale(1.1))\n",
            False,
            None,
            "FadeIn(grid)",
            "grid = grid_initial",
        ),
        (
            "        self.add(grid)\n"
            "        target_grid = Vector(UP)\n"
            "        self.play(ReplacementTransform(grid, target_grid))\n"
            "        grid = target_grid\n"
            "        self.play(grid.animate.scale(1.1))\n",
            True,
            None,
            "Transform(grid, target_grid)",
            "grid = target_grid",
        ),
        (
            "        self.add(grid)\n"
            "        target_grid = Vector(UP)\n"
            "        self.play(ReplacementTransform(grid, target_grid))\n",
            True,
            None,
            "Transform(grid, target_grid)",
            "ReplacementTransform(",
        ),
        (
            "        self.add(grid)\n"
            "        grid_rotated = grid.copy().rotate(PI / 4)\n"
            "        # KD1_ANIMATION_EVENT: grid_event\n"
            "        self.play(Transform(grid_rotated, grid.copy()))\n",
            True,
            "update",
            "Transform(grid, grid.copy())",
            "Transform(grid_rotated,",
        ),
        (
            "        self.play(FadeIn(grid))\n",
            False,
            "introduce",
            "# KD1_ANIMATION_EVENT: grid_event",
            "__auto_",
        ),
    ],
)
def test_inspection_centralizes_existing_real_code_repairs(
    body, active, event, expected, forbidden
):
    from kd1_anime.agents.api_linter import lint_manim_api
    from kd1_anime.agents.continuity import extract_scene_continuity_elements
    from kd1_anime.agents.lifecycle import validate_animation_lifecycle
    from kd1_anime.agents.validator import validate_manim_code
    from kd1_anime.run_store import sha256_text

    plan, technical = vector_contract(active=active, event=event)
    original = vector_candidate(body)
    if "GrowArrow" in body:
        original = original.replace("Vector(RIGHT)", "Arrow3D(ORIGIN, RIGHT)").replace(
            "class Demo(Scene):", "class Demo(ThreeDScene):"
        )
    inspection = CandidateAcceptor().inspect_candidate(original, plan, technical_spec=technical)

    assert inspection.accepted, inspection.feedback
    assert expected in inspection.code
    assert forbidden not in inspection.code
    assert inspection.repairs
    assert inspection.original_code == original
    assert inspection.original_code_sha256 == sha256_text(original)
    assert inspection.code_sha256 == sha256_text(inspection.code)
    assert inspection.code_sha256 != inspection.original_code_sha256
    assert inspection.ast_result == validate_manim_code(inspection.code)
    assert inspection.api_result == lint_manim_api(inspection.code, scene_plan=plan)
    assert inspection.lifecycle_result == validate_animation_lifecycle(inspection.code, technical)
    assert (
        inspection.exported_elements_code
        == extract_scene_continuity_elements(inspection.code, plan)[0]
    )


def test_repairs_api_before_export_alias_and_marker_matching():
    plan, technical = vector_contract(event="introduce")
    code = (
        vector_candidate(
            "        grid_initial = Arrow3D(ORIGIN, RIGHT)\n"
            "        self.play(GrowArrow(grid_initial))\n"
            "        grid = grid_initial\n"
        )
        .replace("Vector(RIGHT)", "Arrow3D(ORIGIN, RIGHT)")
        .replace("class Demo(Scene):", "class Demo(ThreeDScene):")
    )

    inspection = CandidateAcceptor().inspect_candidate(code, plan, technical_spec=technical)

    assert inspection.accepted, inspection.feedback
    assert "self.play(Create(grid))" in inspection.code
    assert "# KD1_ANIMATION_EVENT: grid_event" in inspection.code
    assert "grid = grid_initial" not in inspection.code
    assert len(inspection.repairs) == 3
    assert "GrowArrow" in inspection.repairs[0] and "grid_initial" in inspection.repairs[0]
    assert "grid_initial" in inspection.repairs[1]
    assert "grid_event" in inspection.repairs[2]


def test_inspection_repairs_removed_active_and_revalidates_final_source():
    from kd1_anime.agents.lifecycle import validate_animation_lifecycle

    plan, technical = vector_contract(active=True, removed=True)
    original = vector_candidate("        self.add(grid)\n        self.wait(1)\n", exported=False)
    inspection = CandidateAcceptor().inspect_candidate(original, plan, technical_spec=technical)

    assert inspection.accepted, inspection.feedback
    assert "FadeOut(grid)" in inspection.code
    assert inspection.repairs == ("为仍 active 的移除对象补齐 FadeOut: grid",)
    assert inspection.lifecycle_result == validate_animation_lifecycle(inspection.code, technical)


def test_inspection_strips_only_redundant_optional_exports():
    plan = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="grid", variable_name="grid", required=False)
            ]
        }
    )
    original = vector_candidate("        self.play(FadeIn(grid))\n", exported=False)
    original += (
        "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
        "        # element_id: grid\n"
        "        grid = Vector(RIGHT)\n"
        "        # KD1_CONTINUITY_EXPORT_END\n"
    )

    inspection = CandidateAcceptor().inspect_candidate(original, plan)

    assert inspection.accepted, inspection.feedback
    assert inspection.code.count("grid = Vector(RIGHT)") == 1
    assert "KD1_CONTINUITY_EXPORT_BEGIN" not in inspection.code
    assert inspection.repairs
    readonly = CandidateAcceptor().inspect_candidate(original, plan, allow_repair=False)
    assert readonly.code == original
    assert readonly.repairs == ()


@pytest.mark.parametrize(
    "body",
    [
        "        self.play(ShowCreation(grid))\n",
        "        self.play(FadeIn(grid))\n",
    ],
)
def test_readonly_inspection_rejects_repair_dependent_source_without_changing_it(body):
    from kd1_anime.run_store import sha256_text

    plan, technical = vector_contract(event="introduce")
    original = vector_candidate(body)
    inspection = CandidateAcceptor().inspect_candidate(
        original, plan, technical_spec=technical, allow_repair=False
    )

    assert not inspection.accepted
    assert inspection.code == inspection.original_code == original
    assert inspection.code_sha256 == inspection.original_code_sha256 == sha256_text(original)
    assert not inspection.repairs
    assert inspection.diagnostics
    assert all(item.code_sha256 == inspection.code_sha256 for item in inspection.diagnostics)


def test_custom_validator_cannot_bypass_security_api_or_continuity():
    from kd1_anime.agents.validator import CodeValidationResult

    plan, technical = vector_contract()
    original = vector_candidate("        self.play(FadeIn(grid))\n")
    original = original.replace("grid = Vector(RIGHT)", "grid = Vector(external_value)")
    original += "        self.add(InventedShape())\n        open('/tmp/unsafe', 'w')\n"
    inspection = CandidateAcceptor().inspect_candidate(
        original,
        plan,
        technical_spec=technical,
        manim_version="0.20.1",
        validator=lambda *args, **kwargs: CodeValidationResult(True, scene_classes=["Demo"]),
    )

    assert not inspection.accepted
    assert not inspection.ast_result.is_valid
    assert not inspection.api_result.is_valid
    assert not inspection.continuity_result.is_valid
    assert {item.stage for item in inspection.diagnostics} >= {"ast", "api", "continuity"}


@pytest.mark.parametrize(
    "repair_name",
    [
        "repair_removed_active_lifecycle",
        "repair_missing_animation_markers",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [
        "        open('/tmp/unsafe', 'w')\n",
        "        self.add(InventedShape())\n",
        "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
        "        # element_id: bad\n"
        "        bad = Vector(external_value)\n"
        "        # KD1_CONTINUITY_EXPORT_END\n",
    ],
)
def test_feedback_repairs_cannot_reuse_stale_validation(monkeypatch, repair_name, invalid):
    import kd1_anime.candidate_acceptor as module
    from kd1_anime.run_store import sha256_text

    removed = repair_name == "repair_removed_active_lifecycle"
    plan, technical = vector_contract(
        active=removed, removed=removed, event=None if removed else "introduce"
    )
    original = vector_candidate(
        "        self.add(grid)\n" if removed else "        self.play(FadeIn(grid))\n",
        exported=not removed,
    )
    repair = getattr(module, repair_name)

    def contaminated_repair(*args, **kwargs):
        repaired, repairs = repair(*args, **kwargs)
        assert repairs
        return repaired + invalid, repairs

    monkeypatch.setattr(module, repair_name, contaminated_repair)
    inspection = CandidateAcceptor().inspect_candidate(
        original, plan, technical_spec=technical, manim_version="0.20.1"
    )

    assert not inspection.accepted
    assert inspection.repairs
    assert inspection.code.endswith(invalid)
    assert inspection.code_sha256 == sha256_text(inspection.code)
    assert inspection.diagnostics
    assert all(item.code_sha256 == inspection.code_sha256 for item in inspection.diagnostics)


def test_rejected_acceptance_preserves_inspection_and_does_not_write(tmp_path):
    from kd1_anime.run_store import sha256_text

    destination = tmp_path / "scene.py"
    destination.write_text(CODE)
    original = CODE + "        open('/tmp/unsafe', 'w')\n"
    inspection = CandidateAcceptor().inspect_candidate(original, make_plan())

    with pytest.raises(CandidateRejected) as error:
        CandidateAcceptor().accept_candidate(original, make_plan(), destination=destination)

    assert error.value.inspection == inspection
    assert inspection.code_sha256 == sha256_text(original)
    assert not inspection.accepted
    assert inspection.ast_result.errors
    assert destination.read_text() == CODE


@pytest.mark.parametrize(
    "invalid",
    [
        "        open('/tmp/unsafe', 'w')\n",
        "        self.add(InventedShape())\n",
        "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
        "        # element_id: bad\n"
        "        bad = Vector(external_value)\n"
        "        # KD1_CONTINUITY_EXPORT_END\n",
    ],
)
def test_api_repair_does_not_skip_remaining_validation(invalid):
    plan, technical = vector_contract()
    original = vector_candidate("        self.play(GrowArrow(grid))\n")
    original = (
        original.replace("Vector(RIGHT)", "Arrow3D(ORIGIN, RIGHT)").replace(
            "class Demo(Scene):", "class Demo(ThreeDScene):"
        )
        + invalid
    )

    inspection = CandidateAcceptor().inspect_candidate(
        original, plan, technical_spec=technical, manim_version="0.20.1"
    )

    assert not inspection.accepted
    assert inspection.repairs
    assert "self.play(Create(grid))" in inspection.code
    assert inspection.diagnostics


def test_inspection_never_writes_even_for_successful_repairs(monkeypatch):
    import kd1_anime.candidate_acceptor as module

    def forbidden_write(*args, **kwargs):
        pytest.fail("inspection must not write source files")

    monkeypatch.setattr(module, "atomic_write_text", forbidden_write)
    plan, technical = vector_contract(event="introduce")
    original = vector_candidate("        self.play(FadeIn(grid))\n")

    inspection = CandidateAcceptor().inspect_candidate(original, plan, technical_spec=technical)

    assert inspection.accepted, inspection.feedback
    assert inspection.repairs
    assert inspection.original_code == original


def test_custom_validator_adds_constraints_without_duplicating_builtin_errors():
    from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code

    original = CODE + "        open('/tmp/unsafe', 'w')\n"
    builtin = validate_manim_code(original)
    custom_error = "project-specific math invariant failed"

    def validator(code, **kwargs):
        return CodeValidationResult(
            False, [*validate_manim_code(code, **kwargs).errors, custom_error], ["Demo"]
        )

    inspection = CandidateAcceptor().inspect_candidate(original, make_plan(), validator=validator)

    assert not inspection.accepted
    assert inspection.ast_result.errors == [*builtin.errors, custom_error]
    assert custom_error in inspection.feedback
    assert all(inspection.feedback.count(error) == 1 for error in builtin.errors)


@pytest.mark.parametrize("original", ["", "from manim import *\nclass Demo(Scene):\n    def"])
def test_empty_and_syntax_invalid_sources_return_bound_rejection_evidence(original):
    from kd1_anime.run_store import sha256_text

    inspection = CandidateAcceptor().inspect_candidate(original, make_plan())

    assert not inspection.accepted
    assert inspection.code == original
    assert inspection.original_code_sha256 == inspection.code_sha256 == sha256_text(original)
    assert not inspection.ast_result.is_valid
    assert inspection.diagnostics


def test_independent_direct_inspection_does_not_require_handoff_exports():
    code = (
        "from manim import *\nclass RadiusScene(Scene):\n"
        "    def construct(self):\n"
        "        radius = 1\n"
        "        circle = Circle(radius=radius)\n"
        "        self.add(circle)\n"
    )
    acceptor = CandidateAcceptor()
    assert not acceptor.inspect_candidate(code, make_plan(), allow_repair=False).accepted
    direct = acceptor.inspect_candidate(
        code, make_plan(), allow_repair=False, continuity_mode="none", manim_version="0.20.1"
    )
    assert direct.accepted and direct.code == code
    assert direct.ast_result.is_valid and direct.api_result.is_valid
    assert direct.continuity_result.applicability == "not_applicable"
    assert direct.exported_elements == ()


@pytest.mark.parametrize(
    "invalid", ["        open('/tmp/unsafe', 'w')\n", "        self.add(InventedShape())\n"]
)
def test_direct_inspection_preserves_ast_and_api_boundaries(invalid):
    inspection = CandidateAcceptor().inspect_candidate(
        CODE + invalid,
        make_plan(),
        allow_repair=False,
        continuity_mode="none",
        manim_version="0.20.1",
    )
    assert not inspection.accepted
    assert not inspection.ast_result.is_valid or not inspection.api_result.is_valid


def test_direct_inspection_cannot_disable_explicit_handoff_or_unknown_modes():
    declared = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="circle", variable_name="circle", required=True)
            ]
        }
    )
    with pytest.raises(ValueError):
        CandidateAcceptor().inspect_candidate(CODE, declared, continuity_mode="none")
    with pytest.raises(ValueError):
        CandidateAcceptor().inspect_candidate(CODE, make_plan(), continuity_mode="unknown")
