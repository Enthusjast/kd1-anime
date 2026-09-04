"""为所有 Agent 提供一致的 Manim renderer 能力说明。"""

from typing import Literal

from kd1_anime.config import settings


def renderer_guidance(renderer: Literal["cairo", "opengl"] | None = None) -> str:
    active_renderer = renderer or settings.MANIM_RENDERER
    if active_renderer == "opengl":
        return """## 当前渲染能力
- Renderer: OpenGL。
- OpenGLCamera 没有 `frame`，禁止 `self.camera.frame` 和 `MovingCameraScene` 运镜。
- 需要推近或平移时，对局部 VGroup 使用 Transform/animate，或调整静态布局。
- 不定义自定义 Mobject/VMobject 子类；使用 Manim 标准图形在 construct 中组合，避免 should_render 不兼容。"""
    return """## 当前渲染能力
- Renderer: Cairo。
- 普通 `Scene` 没有可动画的 `self.camera.frame`。
- 需要相机平移或缩放时必须继承 `MovingCameraScene`，之后才可使用 `self.camera.frame.animate`。
- `ThreeDScene` 使用其专用相机 API，不把 MovingCameraScene 的 frame API 混入 3D 场景。"""


def animation_lifecycle_guidance() -> str:
    return """## 动画对象生命周期
- `Create`、`Write`、`FadeIn` 等 introducer 动画会把对象加入场景，不要为了它们预先 `self.add()`。
- 对 `obj.animate`、Transform 的源对象等已有对象执行动画前，确保对象当前仍在场景中。
- `FadeOut`、`ReplacementTransform` 后源对象通常已移除，不要继续对旧引用执行动画。
- 不同时对同一对象施加互相冲突的动画。"""
