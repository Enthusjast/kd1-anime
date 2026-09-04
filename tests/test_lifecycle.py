"""生成代码对象生命周期校验测试。"""

from kd1_anime.agents.lifecycle import validate_animation_lifecycle
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
