"""为所有 Agent 提供一致的 Manim renderer 能力说明。"""

from typing import Literal

from kd1_anime.config import settings


def renderer_guidance(renderer: Literal["cairo", "opengl"] | None = None) -> str:
    active_renderer = renderer or settings.MANIM_RENDERER
    if active_renderer == "opengl":
        return """## 当前渲染能力
- Renderer: OpenGL。
- OpenGLCamera 没有 `frame`，禁止 `self.camera.frame` 和 `MovingCameraScene` 运镜。
- `ThreeDScene`、`ThreeDAxes`、`Surface` 以及 `ThreeDScene.set_camera_orientation(...)` 均受支持；
  三维场景应使用 `ThreeDScene`，固定视角不等于改用普通 `Scene`。
- 需要推近或平移时，对局部 VGroup 使用 Transform/animate，或调整静态布局。
- 不定义自定义 Mobject/VMobject 子类；使用 Manim 标准图形在 construct 中组合，避免 should_render 不兼容。"""
    return """## 当前渲染能力
- Renderer: Cairo。
- 普通 `Scene` 没有可动画的 `self.camera.frame`。
- 需要相机平移或缩放时必须继承 `MovingCameraScene`，之后才可使用 `self.camera.frame.animate`。
- `ThreeDScene` 使用其专用相机 API，不把 MovingCameraScene 的 frame API 混入 3D 场景。"""


def animation_lifecycle_guidance() -> str:
    return """## 动画对象生命周期
- 以 TechnicalSpec 的 `semantic_action` 和 `KD1_ANIMATION_EVENT` 标记为状态合同，
  不以某个具体动画类是否出现在示例中作为能力边界。
- `introduce` 只能让新对象进入场景；`update`/`hold` 的 source 必须已经 active；
  `remove` 后不要继续使用退出对象；`camera` 不改变 Mobject 状态。
- 每个 `self.play` 前写对应的 `# KD1_ANIMATION_EVENT: <event_id>`，未知动画调用
  可以使用，但必须遵守标记事件的对象语义并接受额外 Smoke Render。
- 不同时对同一对象施加互相冲突的动画。"""
