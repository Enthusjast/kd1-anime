# ruff: noqa: F403, F405
from manim import *


class CameraRecipe(MovingCameraScene):
    def construct(self):
        square = Square(color=BLUE)
        self.add(square)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.7).move_to(square), run_time=1)
        self.play(Restore(self.camera.frame), run_time=0.5)
        self.wait(0.5)
