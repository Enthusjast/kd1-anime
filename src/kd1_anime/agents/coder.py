"""
Coder Agent
负责根据 ScenePlan 生成 Manim Python 代码

知识库来源: adithya-s-k/manim_skill 的 22 个规则文件 + 5 个工作示例 + 3 个场景模板
"""

from kd1_anime.agents.base import BaseAgent
from kd1_anime.agents.planner import ScenePlan

# =============================================================================
# Manim Community Edition 完整知识库
# 参考 adithya-s-k/manim_skill 的规则文件体系
# =============================================================================

MANIM_API_KNOWLEDGE = r"""
# Manim Community Edition 完整 API 参考

## 基本结构
```python
from manim import *

class Scene1_Introduction(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template

        # SETUP - 创建对象
        # ANIMATION - 播放动画
        # CLEANUP - 清理场景
        pass
```

命名规范: 一个场景一个类,描述性命名 (如 `Scene1_Introduction`, `Scene2_DerivePDE`)

---

## 场景类型

| 类型 | 用途 | 关键方法 |
|------|------|---------|
| `Scene` | 默认 2D 场景 | `construct()`, `play()`, `wait()` |
| `ThreeDScene` | 3D 场景 | `set_camera_orientation(phi, theta, gamma)` |
| `MovingCameraScene` | 可缩放/平移 | `self.camera.frame.animate.set(width=4)` |

生命周期: `setup()` → `construct()`; 方法: `self.add()`, `self.remove()`, `self.play()`, `self.wait()`

---

## 创建动画
```python
self.play(Create(circle))              # 绘制轮廓
self.play(Write(formula))              # 书写效果 (文字/公式)
self.play(DrawBorderThenFill(shape))   # 先画边框再填充
self.play(FadeIn(obj))                 # 淡入
self.play(FadeIn(obj, shift=UP))       # 从下方滑入
self.play(FadeIn(obj, scale=0.5))      # 从小变大
self.play(FadeOut(obj))                # 淡出
self.play(GrowFromCenter(obj))         # 从中心生长
self.play(GrowFromPoint(obj, ORIGIN))  # 从指定点生长
self.play(GrowFromEdge(obj, DOWN))     # 从边缘生长
self.play(SpinInFromNothing(obj))      # 旋转出现
self.play(Uncreate(obj))               # 反向绘制消失
self.play(AddTextLetterByLetter(text)) # 逐字显示
```

---

## 变换动画
```python
# Transform — 形态变换,源变量保持引用
self.play(Transform(obj1, obj2))

# ReplacementTransform — 替换引用
self.play(ReplacementTransform(obj1, obj2))

# TransformMatchingTex — 按 TeX 子串匹配 (推荐用于公式变换)
self.play(TransformMatchingTex(eq1, eq2))

# TransformMatchingShapes — 按形状匹配
self.play(TransformMatchingShapes(obj1, obj2))

# TransformFromCopy — 保留原对象,复制变换
self.play(TransformFromCopy(original, copy))

# .animate 语法 — 链式调用
self.play(obj.animate.shift(RIGHT).rotate(PI/4).set_color(BLUE))
self.play(obj.animate.scale(2))
self.play(obj.animate.set_color(RED))
self.play(obj.animate.move_to(ORIGIN))
self.play(obj.animate.to_edge(UP))
self.play(obj.animate.next_to(other, RIGHT))

# MoveToTarget — 先设置 target,再动画
obj.target = obj.copy().shift(RIGHT).set_color(RED)
self.play(MoveToTarget(obj))

# path_arc — 弧形路径
self.play(Transform(obj1, obj2), path_arc=PI/2)
```

---

## 动画组合
```python
# 同时播放
self.play(Create(circle), FadeIn(square), run_time=2)

# AnimationGroup — 同时或按 lag_ratio
self.play(AnimationGroup(
    Create(circle),
    FadeIn(square),
    lag_ratio=0.5,  # 0=同时, 0.5=50%重叠, 1=顺序
))

# LaggedStart — 默认 lag_ratio=0.05
self.play(LaggedStart(*[FadeIn(obj) for obj in objs]))

# Succession — 严格顺序
self.play(Succession(
    Create(circle),
    Wait(1),
    FadeOut(circle),
))

# LaggedStartMap — 对组中每个元素应用动画
self.play(LaggedStartMap(FadeIn, group))
```

---

## 动画参数
```python
self.play(anim, run_time=2)           # 时长 (默认 1s)
self.play(anim, rate_func=smooth)     # 缓动函数
```

缓动函数: `smooth` (默认), `linear`, `rush_into`, `rush_from`, `there_and_back`, `double_smooth`, `lingering`
CSS 风格: `ease_in_sine`, `ease_out_bounce`, `ease_in_out_cubic`, `ease_out_elastic`

---

## LaTeX / 数学公式
```python
# 每个 construct() 开头都必须初始化并注册此模板。
tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
tex_template.add_to_preamble(r"\usepackage{ctex}")
config.tex_template = tex_template

# MathTex — 自动进入数学模式
eq = MathTex(r"\frac{d}{dx}f(x) = f'(x)", tex_template=tex_template)

# Tex — 原始 LaTeX
text = Tex(r"This is \textbf{bold}", tex_template=tex_template)

# 多段公式 (用于部分着色/变换)
eq = MathTex("a", "^2", "+", "b", "^2", "=", "c", "^2", tex_template=tex_template)
eq[0].set_color(BLUE)   # "a"
eq[6].set_color(RED)    # "c"

# 按 TeX 文本着色
eq.set_color_by_tex("a", BLUE)
eq.set_color_by_tex("b", RED)

# substrings_to_isolate — 隔离子串用于独立着色
eq = MathTex(
    "x^2 + y^2 = r^2",
    substrings_to_isolate=["x^2", "y^2", "r^2"],
    tex_template=tex_template,
)
eq.set_color_by_tex("x^2", BLUE)
eq.set_color_by_tex("y^2", GREEN)
eq.set_color_by_tex("r^2", YELLOW)

# tex_to_color_map — 一步到位
eq = MathTex(
    r"E = mc^2",
    tex_to_color_map={"E": BLUE, "m": GREEN, "c": YELLOW},
    tex_template=tex_template,
)

# 多行对齐 (&= 对齐等号, \\ 换行)
eq = MathTex(
    r"(a+b)^2", r"&=", r"a^2 + 2ab + b^2", r"\\",
    r"&=", r"a^2", r"+", r"2ab", r"+", r"b^2",
    substrings_to_isolate=["a^2", "2ab", "b^2"],
    tex_template=tex_template,
)
```

---

## 坐标系与绘图
```python
# Axes
axes = Axes(
    x_range=[-3, 3, 1],
    y_range=[-2, 2, 1],
    axis_config={"color": BLUE, "include_tip": True},
)
graph = axes.plot(lambda x: x**2, color=GREEN)
label = axes.get_graph_label(graph, label="y=x^2")

# 坐标转换
point = axes.c2p(1, 2)  # 数学坐标 → 场景坐标

# 参数方程
param = axes.plot_parametric_curve(
    lambda t: np.array([np.cos(t), np.sin(t), 0]),
    t_range=[0, 2*PI],
    color=RED,
)

# 面积
area = axes.get_area(graph, x_range=[-2, 2], color=BLUE, opacity=0.3)

# Riemann 矩形
rects = axes.get_riemann_rectangles(graph, x_range=[0, 3], dx=0.2)

# NumberPlane
plane = NumberPlane()
plane.prepare_for_nonlinear_transform()  # 让变换效果正确

# PolarPlane
polar = PolarPlane()
polar.plot_polar_graph(lambda theta: 1 + np.cos(theta))

# 切线
tangent = axes.get_tangent_line(graph, x=1)

# ValueTracker 动态绘图
tracker = ValueTracker(-3)
dot = always_redraw(lambda: Dot(
    axes.c2p(tracker.get_value(), np.sin(tracker.get_value())),
    color=YELLOW,
))
self.play(tracker.animate.set_value(3), run_time=5)
```

---

## 更新器 (Updaters)
```python
# 基本更新器
dot.add_updater(lambda m: m.next_to(arrow, UP))
self.play(arrow.animate.shift(RIGHT))  # dot 自动跟随

# always_redraw — 每帧重新创建
label = always_redraw(lambda: Tex(
    f"x = {tracker.get_value():.1f}", tex_template=tex_template
).next_to(dot, UP))

# ValueTracker — 动态参数
tracker = ValueTracker(0)
circle = always_redraw(lambda: Circle(radius=tracker.get_value()))
self.play(tracker.animate.set_value(2), run_time=3)

# TracedPath — 轨迹
path = TracedPath(dot.get_center, stroke_color=YELLOW)

# 移除更新器
dot.clear_updaters()
dot.remove_updater(updater_func)

# 物理模拟 (dt 参数)
def spring_update(m, dt):
    # dt = 帧间时间
    m.velocity += -k * m.get_center() * dt
    m.shift(m.velocity * dt)
obj.add_updater(spring_update)
```

---

## 几何图形
```python
circle = Circle(radius=1, color=BLUE, fill_opacity=0.5)
square = Square(side_length=2, color=RED)
rect = Rectangle(width=4, height=2)
line = Line(start=LEFT, end=RIGHT)
arrow = Arrow(start=LEFT, end=RIGHT, buff=0)
dot = Dot(point=ORIGIN, radius=0.08)
arc = Arc(radius=1, start_angle=0, angle=PI/2)
polygon = Polygon(LEFT, RIGHT, UP)
regular = RegularPolygon(n=6)
brace = Brace(obj, direction=DOWN)
brace_text = brace.get_text("说明")

# 样式
circle.set_stroke(color=WHITE, width=3)
circle.set_fill(BLUE, opacity=0.5)
dashed = DashedLine(start, end)
```

---

## 文字
```python
text = Text("Hello", font_size=48, font="Arial")
text = Text("中文", font="Noto Sans CJK SC")  # 中文需指定字体
paragraph = Paragraph("Line 1", "Line 2", alignment="center")
```

---

## 排列与布局
```python
group = VGroup(obj1, obj2, obj3)
group.arrange(RIGHT, buff=0.5)       # 水平排列
group.arrange(DOWN, buff=0.3)        # 垂直排列
group.arrange_in_grid(rows=2, cols=3) # 网格排列
group.move_to(ORIGIN)
group.to_edge(UP, buff=0.5)
group.next_to(other, RIGHT, buff=0.2)
group.center()
group.shift(RIGHT * 2)

# 对齐
group.align_to(other, LEFT)
group.align_on_border(LEFT, buff=0.5)

# SurroundingRectangle
box = SurroundingRectangle(obj, color=YELLOW, buff=0.2)
```

---

## 颜色
```python
# 预定义
RED, GREEN, BLUE, YELLOW, WHITE, ORANGE, PURPLE, PINK, TEAL, GOLD, MAROON
# 变体: _A (最亮) 到 _E (最暗), 如 RED_A, RED_E

# 自定义
color = "#FF5733"
color = ManimColor("#FF5733")

# 渐变
obj.set_color_by_gradient(RED, BLUE, GREEN)
obj.set_color_by_gradient([RED, BLUE])  # 也可以用列表

# 插值
mid_color = interpolate_color(RED, BLUE, alpha=0.5)
```

---

## 3D 场景
```python
class My3DScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(axes))

        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 - v**2),
            u_range=[-2, 2], v_range=[-2, 2],
        )
        self.play(Create(surface))
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        # 固定 2D 覆盖层
        title = Text("3D Scene")
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP)
        self.play(Write(title))
```

3D 对象: `Sphere()`, `Cube()`, `Cone()`, `Cylinder()`, `Torus()`, `Arrow3D()`

---

## MovingCameraScene (缩放/平移)
```python
class ZoomScene(MovingCameraScene):
    def construct(self):
        shapes = VGroup(Circle(), Square(), Triangle()).arrange(RIGHT, buff=2)
        self.add(shapes)

        # 缩放到某个对象
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=4).move_to(shapes[0]))
        self.wait()

        # 平移
        self.play(self.camera.frame.animate.move_to(shapes[2]))
        self.wait()

        # 恢复
        self.play(Restore(self.camera.frame))

        # 跟随对象
        dot = Dot()
        self.camera.frame.add_updater(lambda m: m.move_to(dot))
        self.play(dot.animate.move_to(RIGHT * 5), run_time=3)
        self.camera.frame.clear_updaters()
```

---

## 高亮动画
```python
self.play(Indicate(term))                           # 闪烁高亮
self.play(Circumscribe(equation, color=YELLOW))     # 圈出
self.play(FlashAround(result))                      # 闪光环绕
self.play(Flash(dot, color=RED))                    # 闪光点
self.play(FocusOn(dot))                             # 聚焦
```

---

## 常见陷阱 (必须避免)

1. **版本混淆**: 使用 `from manim import *` (Community), 不要 `from manimlib import *` (3b1b)
2. **废弃 API**: 用 `Create` 代替 `ShowCreation`, 用 `Tex`/`MathTex` 代替 `TextMobject`
3. **LaTeX 转义**: 数学公式必须用 raw string 写反斜杠命令. `r"\frac{a}{b}"` (raw) 与 `"\\frac{a}{b}"` (普通字符串) 等价, LaTeX 中都是 `\frac`. 千万不要写 `"\frac{a}{b}"` (单反斜杠普通字符串) 或 `"\\frac..."` 之外的形式. 注意 `{` `}` 在 f-string 中需双写为 `{{` `}}`
4. **Transform vs ReplacementTransform**: Transform 保持源变量引用; ReplacementTransform 替换引用
5. **对象超出画面**: 使用 `.to_edge()`, `.move_to()`, `.next_to()` 定位,避免坐标超出 [-7,7]×[-4,4]
6. **中文文字**: 使用 `Text("中文", font="Noto Sans CJK SC")`, 需要系统安装对应字体
7. **动画时长**: 单个 `self.play()` 默认 1 秒; 复杂动画用 `run_time=2` 或更长
8. **不要包含 `if __name__`**: 代码只包含场景类定义

---

## 动画时长参考

| 动作 | 典型时长 |
|------|---------|
| 简单形状创建 | 0.5-1s |
| 文字/公式书写 | 1-2s |
| 变换 | 1-2s |
| 摄像机移动 | 2-3s |
| 吸收停顿 | 0.5-1s |
| 复杂动画 | 2-4s |

---

## 场景模板

### 基本 2D 场景
```python
from manim import *

class YourScene(Scene):
    def construct(self):
        # SETUP
        title = Text("Your Title", font_size=48)
        shape = Circle(color=BLUE, fill_opacity=0.5)
        title.to_edge(UP)
        shape.move_to(ORIGIN)

        # ANIMATION
        self.play(Write(title))
        self.wait(0.5)
        self.play(Create(shape))
        self.wait(0.5)
        self.play(shape.animate.scale(1.5).set_color(RED))
        self.wait()

        # CLEANUP
        self.play(FadeOut(title), FadeOut(shape))
        self.wait()
```

### 公式推导场景
```python
from manim import *

class EquationDerivation(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template

        title = Tex(
            r"Deriving the Quadratic Formula",
            font_size=42,
            tex_template=tex_template,
        )
        title.to_edge(UP)
        self.play(Write(title))

        eq1 = MathTex(r"ax^2 + bx + c = 0", tex_template=tex_template)
        self.play(Write(eq1))
        self.wait(0.5)

        eq2 = MathTex(r"x^2 + \frac{b}{a}x + \frac{c}{a} = 0",
                       tex_to_color_map={r"\frac{b}{a}": BLUE, r"\frac{c}{a}": GREEN},
                       tex_template=tex_template)
        self.play(TransformMatchingTex(eq1, eq2))
        self.wait(0.5)

        eq3 = MathTex(r"\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2}",
                       tex_to_color_map={r"b^2 - 4ac": YELLOW},
                       tex_template=tex_template)
        self.play(TransformMatchingTex(eq2, eq3))
        self.wait(0.5)

        result = MathTex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
                          tex_to_color_map={r"b^2 - 4ac": YELLOW},
                          tex_template=tex_template)
        box = SurroundingRectangle(result, color=YELLOW)
        self.play(TransformMatchingTex(eq3, result))
        self.play(Create(box))
        self.wait(2)
```

### 图表动画场景
```python
from manim import *

class GraphAnimation(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template

        axes = Axes(x_range=[-3, 3], y_range=[-2, 2],
                     axis_config={"color": BLUE})
        graph = axes.plot(lambda x: np.sin(x), color=GREEN)
        label = axes.get_graph_label(graph, label="y=\\sin(x)")

        self.play(Create(axes))
        self.play(Create(graph), Write(label))
        self.wait()

        area = axes.get_area(graph, x_range=[0, PI], color=BLUE, opacity=0.3)
        self.play(FadeIn(area))
        self.wait(2)
```
"""

CODER_SYSTEM_PROMPT = f"""你是一个 Manim 动画编程专家.你的任务是根据导演分镜编写高质量的 Manim Python 代码.

## 你的角色

你收到的不是伪代码,而是导演的视觉设计描述 (画面设计/运镜/流程/关键时刻/数学规格).
你需要自己判断用哪些 Manim 类 (Axes/Dot/Circle/MathTex/ParametricFunction 等)
和动画方法 (FadeIn/Transform/Write/MoveAlongPath 等) 来最好地实现导演的意图.

## 核心要求

1. 必须继承 `Scene` 类 (或 `ThreeDScene`/`MovingCameraScene`),实现 `construct` 方法
2. 使用 `from manim import *` 导入
3. 代码必须可直接渲染,不要包含任何解释性文字
4. 只输出纯 Python 代码,包裹在 ```python ``` 中
5. 视觉效果要丰富、流畅,避免单调的文字展示
6. 数学公式使用 `MathTex`; 中文文字优先使用 `Text(..., font="Noto Sans CJK SC")`
7. 只允许导入 manim、numpy、math 和纯计算型标准库; 禁止文件、网络、shell、subprocess、eval/exec
8. **必须**在每个 `construct()` 方法开头添加以下模板代码（不可省略）:
```python
tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
tex_template.add_to_preamble(r"\\usepackage{ctex}")
config.tex_template = tex_template
```
9. **必须**在每个 `Tex`/`MathTex` 调用中显式传入 `tex_template=tex_template`; 禁止依赖默认的 latex/pdflatex
10. 中文文本必须使用 `Tex(r"中文", tex_template=tex_template)` 而不是 `Text()`

## ⚠️ 特别注意：TexTemplate 是强制要求
如果你的代码包含任何 Tex 或 MathTex，但没有正确配置 TexTemplate，代码将无法通过校验！

## 视觉设计原则

1. **Progressive Disclosure** — 永远不要一次展示所有内容,逐步构建复杂度
2. **Transform, Don't Replace** — 用 `TransformMatchingTex` 而非 FadeOut/FadeIn,保持视觉连续性
3. **Color as Meaning** — 颜色编码一致: 已知=BLUE, 结果=GREEN, 高亮=YELLOW, 错误=RED
4. **Spatial Relationships** — 左→右=变换/时间, 上→下=层级, 中心=焦点, 边缘=上下文

## 动画节奏

遵循 快-快-慢 模式:
- 简单形状: 0.5-1s
- 文字/公式: 1-2s
- 变换: 1-2s
- 复杂动画: 2-4s
- 吸收停顿: 0.5-1s

## Manim API 完整参考

{MANIM_API_KNOWLEDGE}
"""


class CoderAgent(BaseAgent):
    """代码生成 Agent"""

    name = "Coder"

    def generate_code(
        self,
        scene_plan: ScenePlan,
        feedback: str = "",
        previous_code: str = "",
        *,
        stream: bool = True,
    ) -> str:
        """
        根据场景规划生成 Manim 代码

        Args:
            scene_plan: 场景规划
            feedback: 可选的 Reviewer 反馈,用于修正
            previous_code: 上一版代码 (供修正时参考)

        Returns:
            生成的 Python 代码字符串
        """
        self._log(f"正在为 Scene {scene_plan.scene_id} [{scene_plan.title}] 生成代码...")

        user_msg = f"""## 场景导演分镜

- **Scene ID**: {scene_plan.scene_id}
- **标题**: {scene_plan.title}
- **预估时长**: {scene_plan.duration_seconds} 秒
- **叙事作用**: {scene_plan.purpose}
- **数学概念**: {scene_plan.math_concept}

### 画面设计
{scene_plan.visual_design}

### 运镜方案
{scene_plan.camera_movement}

### 视觉流程
{chr(10).join(f"- {step}" for step in scene_plan.visual_flow)}

### 关键时刻
{chr(10).join(f"- {moment}" for moment in scene_plan.key_moments)}

### 数学/物理规格
{scene_plan.computation}
"""

        if previous_code:
            user_msg += f"""
## 上一版代码 (请在此基础上修改, 保留好的部分)

```python
{previous_code}
```
"""

        if feedback:
            user_msg += f"""
## Reviewer 反馈 (请根据以下反馈修正代码)

{feedback}
"""

        user_msg += "\n请生成完整的 Manim Python 代码:"

        code = self.call_llm(
            system_prompt=CODER_SYSTEM_PROMPT,
            user_message=user_msg,
            stream=stream,
        )

        extracted = self._extract_code_block(code)
        self._log(f"代码生成完成 ({len(extracted)} 字符)")
        return extracted
