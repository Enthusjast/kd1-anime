# ruff: noqa: F403, F405
from manim import *


class UpdaterRecipe(Scene):
    def construct(self):
        tracker = ValueTracker(0)
        dot = always_redraw(lambda: Dot(RIGHT * tracker.get_value(), color=YELLOW))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1, rate_func=smooth)
        dot.clear_updaters()
        self.wait(0.5)
