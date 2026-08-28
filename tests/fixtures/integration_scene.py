"""最小真实渲染探针；只用于手动集成 workflow，不参与普通单元测试。"""

from manim import DOWN, MathTex, Scene, Tex, TexTemplate, Write, config


class IntegrationScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template

        title = Tex(r"真实渲染探针", tex_template=tex_template)
        formula = MathTex(r"a^2+b^2=c^2", tex_template=tex_template)
        formula.next_to(title, DOWN)
        self.play(Write(title), run_time=0.2)
        self.play(Write(formula), run_time=0.2)
        self.wait(0.1)
