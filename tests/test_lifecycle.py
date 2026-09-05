"""生成代码对象生命周期校验测试。"""

from kd1_anime.agents.lifecycle import (
    detect_unknown_animations,
    repair_removed_active_lifecycle,
    repair_required_export_alias_lifecycle,
    repair_required_export_replacement_lifecycle,
    repair_required_export_transform_alias_lifecycle,
    validate_animation_lifecycle,
)
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec


def spec(*, exported=("formula",), initially_active=()):
    return TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id=name,
                variable_name=name,
                constructor="MathTex",
                initially_active=name in initially_active,
            )
            for name in {"formula", "step"}
        ],
        export_element_ids=list(exported),
    )


def test_lifecycle_accepts_fade_in_and_animate():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        self.play(FadeIn(formula))
        self.play(formula.animate.scale(1.1))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is True


def test_repairs_required_export_alias_after_temporary_intro():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="v1",
                variable_name="v1",
                constructor="Vector",
                exported=True,
            ),
            TechnicalObject(
                element_id="v2",
                variable_name="v2",
                constructor="Vector",
                exported=True,
            ),
        ],
        export_element_ids=["v1", "v2"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: v1
        v1 = Vector([1, 1])
        # element_id: v2
        v2 = Vector([1, -1])
        # KD1_CONTINUITY_EXPORT_END
        v1_initial = Vector([1, 1])
        v2_initial = Vector([1, -1])
        self.play(FadeIn(v1_initial), FadeIn(v2_initial))
        v1_scaled = Vector([4, 4])
        v2_scaled = Vector([2, -2])
        self.play(Transform(v1_initial, v1_scaled), Transform(v2_initial, v2_scaled))
        v1 = v1_initial
        v2 = v2_initial
        self.play(v1.animate.set_stroke(width=6), v2.animate.set_stroke(width=6))
"""

    repaired, repairs = repair_required_export_alias_lifecycle(code, technical_spec)

    assert repairs
    assert "v1 = v1_initial" not in repaired
    assert "FadeIn(v1)" in repaired
    assert "Transform(v1, v1_scaled)" in repaired
    result = validate_animation_lifecycle(repaired, technical_spec)
    assert result.is_valid is True, result.errors


def test_does_not_rewrite_unrelated_python_alias():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="MathTex",
                exported=True,
            )
        ],
        export_element_ids=["formula"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x")
        # KD1_CONTINUITY_EXPORT_END
        formula_initial = formula.copy()
        value = formula_initial
        self.play(FadeIn(formula))
"""

    repaired, repairs = repair_required_export_alias_lifecycle(code, technical_spec)

    assert repaired == code
    assert repairs == ()


def test_repairs_replacement_target_rebound_to_required_export():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="grid",
                variable_name="grid",
                constructor="NumberPlane",
                initially_active=True,
                exported=True,
            )
        ],
        export_element_ids=["grid"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: grid
        grid = NumberPlane()
        # KD1_CONTINUITY_EXPORT_END
        self.add(grid)
        target_grid = NumberPlane()
        self.play(ReplacementTransform(grid, target_grid))
        grid = target_grid
        self.play(grid.animate.scale(0.9))
"""

    repaired, repairs = repair_required_export_transform_alias_lifecycle(code, technical_spec)

    assert repairs
    assert "ReplacementTransform(grid, target_grid)" not in repaired
    assert "Transform(grid, target_grid)" in repaired
    assert "grid = target_grid" not in repaired
    assert "self.play(grid.animate.scale(0.9))" in repaired
    result = validate_animation_lifecycle(repaired, technical_spec)
    assert result.is_valid is True, result.errors


def test_repairs_reported_active_removed_objects_at_construct_end():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="grid",
                variable_name="grid",
                constructor="NumberPlane",
                initially_active=True,
            ),
            TechnicalObject(
                element_id="vector",
                variable_name="vector",
                constructor="Vector",
                initially_active=True,
            ),
        ],
        removed_element_ids=["grid", "vector"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        grid = NumberPlane()
        vector = Vector(RIGHT)
        self.add(grid, vector)
        self.wait(1)
"""

    repaired, repairs = repair_removed_active_lifecycle(
        code,
        technical_spec,
        ["场景结束时已移除对象仍 active: grid, vector"],
    )

    assert repairs == ("为仍 active 的移除对象补齐 FadeOut: grid, vector",)
    assert "self.play(FadeOut(grid, vector), run_time=0.5)" in repaired
    result = validate_animation_lifecycle(repaired, technical_spec)
    assert result.is_valid is True, result.errors


def test_repairs_replacement_of_required_export_without_rebinding():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="grid",
                variable_name="grid",
                constructor="NumberPlane",
                initially_active=True,
                exported=True,
            )
        ],
        export_element_ids=["grid"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: grid
        grid = NumberPlane()
        # KD1_CONTINUITY_EXPORT_END
        self.add(grid)
        target_grid = NumberPlane()
        self.play(ReplacementTransform(grid, target_grid))
        self.wait(1)
"""

    repaired, repairs = repair_required_export_replacement_lifecycle(code, technical_spec)

    assert repairs
    assert "Transform(grid, target_grid)" in repaired
    result = validate_animation_lifecycle(repaired, technical_spec)
    assert result.is_valid is True, result.errors


def test_repairs_replacement_of_required_export_to_declared_target():
    technical_spec = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula_before",
                variable_name="formula_before",
                constructor="MathTex",
                initially_active=True,
                exported=True,
            ),
            TechnicalObject(
                element_id="formula_after",
                variable_name="formula_after",
                constructor="MathTex",
            ),
        ],
        export_element_ids=["formula_before"],
    )
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula_before
        formula_before = MathTex(r"x")
        # KD1_CONTINUITY_EXPORT_END
        formula_after = MathTex(r"x^2")
        self.add(formula_before)
        self.play(ReplacementTransform(formula_before, formula_after))
"""

    repaired, repairs = repair_required_export_replacement_lifecycle(code, technical_spec)

    assert repairs
    assert "Transform(formula_before, formula_after)" in repaired
    result = validate_animation_lifecycle(repaired, technical_spec)
    assert result.is_valid is True, result.errors


def test_lifecycle_accepts_animation_of_loop_variable():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formulas = [MathTex(r"x"), MathTex(r"x^2")]
        for formula in formulas:
            self.play(Write(formula))
    """

    result = validate_animation_lifecycle(code, spec(exported=()))

    assert result.is_valid is True


def test_lifecycle_ignores_threedscene_camera_animation():
    code = """
from manim import *
class Demo(ThreeDScene):
    def construct(self):
        formula = MathTex(r"z=f(x,y)")
        self.play(FadeIn(formula))
        self.play(self.camera.animate.set_euler_angles(theta=1.2, phi=0.8))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is True, result.errors


def test_lifecycle_tracks_vgroup_aliases_for_member_objects():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        step = MathTex(r"x^2")
        formula_group = VGroup(formula, step)
        self.play(FadeIn(formula_group))
        self.play(formula.animate.scale(1.1))
        self.play(FadeOut(formula_group))
    """

    result = validate_animation_lifecycle(code, spec(exported=()))

    assert result.is_valid is True


def test_lifecycle_rejects_transform_source_that_is_not_active():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        step = MathTex(r"x^2")
        self.play(Transform(formula, step))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is False
    assert any("source 未 active" in error for error in result.errors)


def test_lifecycle_rejects_using_transform_target_as_active_object():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        step = MathTex(r"x^2")
        self.play(FadeIn(formula))
        self.play(Transform(formula, step))
        self.play(step.animate.scale(1.1))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is False
    assert any("animate 作用于未 active" in error for error in result.errors)


def test_lifecycle_rejects_fade_out_after_replacement_transform():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        step = MathTex(r"x^2")
        self.play(FadeIn(formula))
        self.play(ReplacementTransform(formula, step))
        self.play(FadeOut(formula))
"""

    result = validate_animation_lifecycle(code, spec(exported=("step",)))

    assert result.is_valid is False
    assert any("未 active" in error for error in result.errors)


def test_lifecycle_tracks_matching_tex_as_an_in_place_transform():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        next_formula = MathTex(r"x^2")
        self.play(FadeIn(formula))
        self.play(TransformMatchingTex(formula, next_formula))
        self.play(formula.animate.scale(1.1))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is True


def test_lifecycle_tracks_multiple_fade_out_arguments():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        step = MathTex(r"x^2")
        self.play(FadeIn(formula), FadeIn(step))
        self.play(FadeOut(formula, step))
"""

    result = validate_animation_lifecycle(code, spec(exported=()))

    assert result.is_valid is True


def test_lifecycle_accepts_common_grow_introducer_and_indicate():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        self.play(GrowFromCenter(formula))
        self.play(Indicate(formula))
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is True


def test_lifecycle_rejects_clearing_required_export():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        self.play(FadeIn(formula))
        self.clear()
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is False
    assert any("self.clear" in error for error in result.errors)


def test_lifecycle_allows_initially_active_inherited_element_to_fade_out():
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        self.play(FadeOut(formula))
"""

    result = validate_animation_lifecycle(
        code,
        spec(exported=("step",), initially_active=("formula",)),
    )

    assert result.is_valid is False  # step is required but never created
    assert any("必须导出" in error for error in result.errors)


def test_lifecycle_rejects_scene_side_effect_hidden_in_helper():
    code = """
from manim import *
class Demo(Scene):
    def show_formula(self, formula):
        self.play(FadeIn(formula))

    def construct(self):
        formula = MathTex(r"x")
        self.show_formula(formula)
"""

    result = validate_animation_lifecycle(code, spec())

    assert result.is_valid is False
    assert any("辅助函数" in error and "self.play" in error for error in result.errors)


def _semantic_spec(action="introduce"):
    return TechnicalSpec(
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
                "semantic_action": action,
                "target_element_ids": ["formula"] if action == "introduce" else [],
                "create_element_ids": ["formula"] if action == "introduce" else [],
                "source_element_ids": ["formula"] if action != "introduce" else [],
            }
        ],
        export_element_ids=["formula"],
    )


def test_semantic_marker_allows_an_animation_not_in_static_name_list():
    technical_spec = _semantic_spec()
    code = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        # KD1_ANIMATION_EVENT: show_formula
        self.play(CustomReveal(formula))
"""

    result = validate_animation_lifecycle(code, technical_spec)

    assert result.is_valid is True, result.errors
    assert result.unknown_animations
    assert any("unknown-animation" in warning for warning in result.warnings)
    assert detect_unknown_animations(code, technical_spec) == result.unknown_animations


def test_semantic_marker_is_required_and_must_reference_contract_event():
    technical_spec = _semantic_spec()
    missing_marker = """
from manim import *
class Demo(Scene):
    def construct(self):
        formula = MathTex(r"x")
        self.play(FadeIn(formula))
"""
    unknown_marker = missing_marker.replace(
        "self.play", "# KD1_ANIMATION_EVENT: wrong\n        self.play"
    )

    missing_result = validate_animation_lifecycle(missing_marker, technical_spec)
    unknown_result = validate_animation_lifecycle(unknown_marker, technical_spec)

    assert not missing_result.is_valid
    assert any("缺少语义事件标记" in error for error in missing_result.errors)
    assert not unknown_result.is_valid
    assert any("未在 TechnicalSpec 中声明" in error for error in unknown_result.errors)


def test_camera_semantic_event_does_not_require_mobject_lifecycle():
    technical_spec = TechnicalSpec(
        scene_id=1,
        animations=[
            {
                "event_id": "camera_move",
                "start_seconds": 0,
                "end_seconds": 1,
                "semantic_action": "camera",
            }
        ],
    )
    code = """
from manim import *
class Demo(ThreeDScene):
    def construct(self):
        # KD1_ANIMATION_EVENT: camera_move
        self.play(self.camera.animate.set_euler_angles(theta=1.0, phi=0.8))
"""

    result = validate_animation_lifecycle(code, technical_spec)

    assert result.is_valid is True, result.errors


def test_marker_cannot_introduce_an_object_not_used_by_animation():
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        formula = Circle()
        other = Square()
        # KD1_ANIMATION_EVENT: show_formula
        self.play(FadeIn(other))
"""
    result = validate_animation_lifecycle(code, _semantic_spec())
    assert not result.is_valid


def test_update_marker_cannot_hide_actual_removal():
    technical = _semantic_spec("update")
    technical.objects[0].initially_active = True
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        formula = Circle()
        self.add(formula)
        # KD1_ANIMATION_EVENT: show_formula
        self.play(FadeOut(formula))
"""
    assert not validate_animation_lifecycle(code, technical).is_valid
