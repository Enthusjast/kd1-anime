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
# Manim Community Edition API 参考 (精简版)

## 基本结构
```python
from manim import *

class MyScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template
        # 代码...
```

## 常用场景类型
- `Scene`: 默认 2D
- `ThreeDScene`: 3D 场景
- `MovingCameraScene`: 可缩放/平移

## 常用动画
```python
self.play(Create(obj))           # 绘制
self.play(Write(text))           # 书写
self.play(FadeIn(obj))           # 淡入
self.play(FadeOut(obj))          # 淡出
self.play(Transform(a, b))       # 变换
self.play(ReplacementTransform(a, b))  # 替换变换
self.play(obj.animate.shift(RIGHT))    # 移动
self.play(obj.animate.scale(2))        # 缩放
self.play(obj.animate.set_color(RED))  # 变色
```

## 常用对象
```python
circle = Circle(radius=1, color=BLUE)
square = Square(side_length=2)
line = Line(start, end)
arrow = Arrow(start, end)
dot = Dot(point=ORIGIN)
text = Text("Hello", font="Noto Sans CJK SC")
tex = Tex(r"E=mc^2", tex_template=tex_template)
math = MathTex(r"\frac{a}{b}", tex_template=tex_template)
axes = Axes(x_range=[-3,3], y_range=[-2,2])
graph = axes.plot(lambda x: x**2)
```

## 布局
```python
obj.next_to(other, RIGHT, buff=0.5)
obj.to_edge(UP)
obj.move_to(ORIGIN)
obj.shift(RIGHT * 2)
group = VGroup(obj1, obj2)
group.arrange(RIGHT, buff=0.5)
```

## 颜色
RED, GREEN, BLUE, YELLOW, WHITE, ORANGE, PURPLE, PINK
color="#FF6B6B" (自定义)

## 3D
```python
axes = ThreeDAxes()
sphere = SurfaceSphere()
self.set_camera_orientation(phi=75*DEGREES, theta=30*DEGREES)
```

## 提示
- 每个场景一个类
- 使用 tex_template 配置 XeLaTeX
- 中文用 Text(..., font="Noto Sans CJK SC")
"""

CODER_SYSTEM_PROMPT = r"""你是一个 Manim 动画编程专家.你的任务是根据导演分镜编写高质量的 Manim Python 代码.

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
tex_template.add_to_preamble(r"\usepackage{ctex}")
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

""" + MANIM_API_KNOWLEDGE


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
