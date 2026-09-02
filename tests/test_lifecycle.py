"""生成代码对象生命周期校验测试。"""

from kd1_anime.agents.lifecycle import (
    repair_required_export_alias_lifecycle,
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
