# ruff: noqa: F403, F405
from manim import *


class GeometryRecipe(Scene):
    def construct(self):
        triangle = Polygon(LEFT * 2, RIGHT * 2, UP * 2, color=BLUE)
        label = Text("triangle", font_size=24).next_to(triangle, DOWN)
        self.play(Create(triangle), Write(label), run_time=1)
        self.wait(0.5)
