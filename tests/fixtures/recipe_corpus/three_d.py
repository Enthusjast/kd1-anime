# ruff: noqa: F403, F405
import numpy as np
from manim import *


class ThreeDRecipe(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes()
        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(10, 10),
            fill_opacity=0.5,
        )
        self.play(Create(axes), Create(surface), run_time=1)
        self.wait(0.5)
