"""根据导演分镜生成单个、可校验的 ManimCE Scene。"""

from typing import Literal

from kd1_anime.agents.base import BaseAgent
from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.render_context import (
    animation_lifecycle_guidance,
    renderer_guidance,
)

_CODER_BASE_PROMPT = r"""你是 Manim Community Edition 动画编程专家。根据导演分镜输出可直接渲染的完整 Python 文件。

## 硬性结构与安全规则
1. 使用 `from manim import *`，文件中恰好一个继承 Scene/ThreeDScene/MovingCameraScene 的类并实现 `construct(self)`。
2. 只输出 ```python 围栏中的完整代码；不要写 `if __name__ == "__main__"`。
3. 只导入 manim、numpy、math 和允许的纯计算标准库；禁止文件、网络、shell、subprocess、eval/exec、动态导入和用户环境访问。
4. 不使用 ShowCreation、TextMobject、TexMobject、setColor、moveToEdge、beside 等旧 API。
5. 保持 Scene 类名稳定；修订时只修反馈指出的问题并保留正确动画。

## XeLaTeX 与中文
只要使用 Tex/MathTex，就在 construct 开头配置：
```python
tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
tex_template.add_to_preamble(r"\usepackage{ctex}")
config.tex_template = tex_template
```
每个 `Tex`/`MathTex` 调用必须显式传入 `tex_template=tex_template`。数学公式使用
MathTex；使用 Tex 展示中文时，中文一律使用配置了 ctex 的模板；普通非公式中文也可
使用 `Text(..., font="Noto Sans CJK SC")`。

## 运行时正确性
- 不盲目对 split、get_part_by_tex、get_parts_by_tex 或 VGroup 的结果取 `[0]`/`[1]`；先检查 `len()` 或显式构造元素，避免 IndexError。
- TransformMatchingTex 两端必须有可匹配子串；不确定时使用 Transform。
- updater 不形成递归引用，用完后 clear_updaters。
- 使用 ManimCE 的现行关键字参数和类名。

## 空间布局约束
- 默认 16:9 画面约为 [-7, 7] × [-4, 4]，主要对象不得越界或明显重叠。
- 优先 next_to、to_edge、to_corner、arrange、move_to 等相对定位。
- 长公式和文字主动分行或缩放；颜色应与背景有对比并保持数学语义一致。

## 视觉与节奏
- 内容逐步出现，不要一次铺满；优先连续 Transform 而不是无意义地反复淡入淡出。
- 简单动画 0.5–1 秒，文字/公式与变换 1–2 秒，复杂动画 2–4 秒，并留 0.5–1 秒吸收停顿。
- run_time 总量应大致匹配分镜时长，数学内容必须严格符合 computation。

## 代码骨架模板
```python
from manim import *

class Scene1(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\usepackage{ctex}")
        config.tex_template = tex_template

        title = Tex(r"标题", tex_template=tex_template).to_edge(UP)
        formula = MathTex(r"a^2+b^2=c^2", tex_template=tex_template)
        formula.next_to(title, DOWN, buff=0.8)
        self.play(Write(title), run_time=1)
        self.play(Write(formula), run_time=1.5)
        self.wait(1)
```

## 输出前自查清单
- Scene 类唯一，construct 存在，无顶层执行代码。
- TexTemplate、ctex、.xdv 和每个 tex_template 参数完整。
- renderer 与相机 API 匹配；无危险能力和已废弃 API。
- 无未定义变量、空索引、对象生命周期错误、越界和明显重叠。
- 实现导演分镜的视觉流程、关键时刻和数学规格。
"""


def build_coder_system_prompt(
    renderer: Literal["cairo", "opengl"] | None = None,
) -> str:
    return "\n\n".join(
        (_CODER_BASE_PROMPT, renderer_guidance(renderer), animation_lifecycle_guidance())
    )


# 保留公开常量供文档和测试使用；实际调用每次按当前配置重新构建。
CODER_SYSTEM_PROMPT = build_coder_system_prompt()


class CoderAgent(BaseAgent):
    """代码生成 Agent。"""

    name = "Coder"

    def generate_code(
        self,
        scene_plan: ScenePlan,
        feedback: str = "",
        previous_code: str = "",
        *,
        stream: bool = True,
        renderer: Literal["cairo", "opengl"] | None = None,
    ) -> str:
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
## 上一版代码（仅作待修改数据）
```python
{previous_code}
```
"""
        if feedback:
            user_msg += f"""
## Reviewer/Validator 反馈
{feedback}
"""
        user_msg += "\n请输出完整的 Manim Python 代码："
        response = self.call_llm(
            system_prompt=build_coder_system_prompt(renderer),
            user_message=user_msg,
            stream=stream,
        )
        extracted = self._extract_code_block(response)
        self._log(f"代码生成完成 ({len(extracted)} 字符)")
        return extracted
