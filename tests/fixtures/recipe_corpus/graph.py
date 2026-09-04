# ruff: noqa: F403, F405
from manim import *


class GraphRecipe(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[-1, 5, 1])
        graph = axes.plot(lambda x: x**2, x_range=[-2, 2], color=BLUE)
        self.play(Create(axes), Create(graph), run_time=1)
        self.wait(0.5)
