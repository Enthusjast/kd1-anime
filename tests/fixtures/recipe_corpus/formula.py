# ruff: noqa: F403, F405
from manim import *


class FormulaRecipe(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template
        formula = MathTex(r"a^2+b^2=c^2", tex_template=tex_template, color=YELLOW)
        result = MathTex(r"c^2=a^2+b^2", tex_template=tex_template, color=YELLOW)
        self.play(Write(formula))
        self.play(Transform(formula, result), run_time=1)
        self.wait(0.5)
