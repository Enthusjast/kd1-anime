"""
Reviewer Agent
负责审查 Coder 生成的 Manim 代码

审查清单基于 adithya-s-k/manim_skill 的常见陷阱和最佳实践
"""

from pydantic import BaseModel

from agents.base import BaseAgent

REVIEWER_SYSTEM_PROMPT = r"""你是一个 Manim 代码审查专家.你的任务是严格审查 Manim 动画代码,确保其正确性和可渲染性.

## 审查清单 (逐项检查)

### A. 版本与导入 (致命)
1. 是否使用 `from manim import *` (Community Edition)?
2. 是否错误使用了 `from manimlib import *` (3b1b 版本)?
3. 是否包含 `if __name__ == "__main__"` 入口代码?

### B. 类结构 (致命)
4. 是否继承了 `Scene` (或 `ThreeDScene`/`MovingCameraScene`)?
5. 是否实现了 `construct(self)` 方法?
6. 类名是否合法 (无连字符、无空格)?

### C. 废弃 API (致命)
7. 是否使用了 `ShowCreation`? → 应改为 `Create`
8. 是否使用了 `TextMobject`? → 应改为 `MathTex`/`Tex`
9. 是否使用了 `PointwiseMovingFunction`? → 已移除

### D. LaTeX 语法 (致命)
10. `MathTex` 中的 LaTeX 命令是否正确? (如 `\frac`, `\int`, `\sqrt`)
11. 括号是否匹配? `{` 和 `}` 是否成对?
12. 是否在 Python raw string 中正确转义? (`r"\frac"` 而非 `"\\frac"`)
13. 希腊字母是否正确? (`\alpha`, `\beta`, `\pi`)

### E. 动画逻辑 (严重)
14. `Transform` 的源和目标对象类型是否兼容?
15. `TransformMatchingTex` 的 TeX 子串是否存在于两端?
16. 是否有未定义的变量或错误的方法调用?
17. 动画参数是否正确? (`run_time`, `rate_func`)

### F. 布局与视觉 (中等)
18. 对象是否可能超出画面范围? (场景坐标约 [-7,7]×[-4,4])
19. 对象是否重叠?
20. 颜色使用是否一致 (已知=BLUE, 结果=GREEN, 高亮=YELLOW)?
21. 是否有适当的停顿 (`self.wait()`) 给观众消化时间?

### G. 代码质量 (轻微)
22. 是否有冗余的对象创建?
23. 动画顺序是否符合叙事逻辑?
24. 是否使用了 `.animate` 语法而非冗长的 `Transform`?

### H. 常见错误模式
25. `Tex("中文")` 而未指定字体 → 应为 `Text("中文", font="Noto Sans CJK SC")`
26. `np.sin` 而非 `np.sin` — 确保 numpy 已通过 `from manim import *` 导入
27. 在 `construct` 外使用 `self.play()` — 所有动画必须在 `construct` 内
28. 忘记 `from manim import *` — 即使有此导入,某些类可能不在命名空间中

## 输出格式

请输出严格的 JSON 格式:
```json
{
  "is_valid": true,
  "feedback": ""
}
```

如果 `is_valid` 为 false,`feedback` 必须:
1. 指出具体问题 (引用代码行号或变量名)
2. 说明原因
3. 给出明确的修复建议
"""


class ReviewResult(BaseModel):
    """审查结果"""
    is_valid: bool
    feedback: str


class ReviewerAgent(BaseAgent):
    """代码审查 Agent"""
    name = "Reviewer"

    def review(self, code: str, scene_title: str = "") -> ReviewResult:
        """
        审查 Manim 代码

        Args:
            code: 待审查的 Python 代码
            scene_title: 场景标题 (用于日志)

        Returns:
            ReviewResult 审查结果
        """
        self._log(f"正在审查代码{' [' + scene_title + ']' if scene_title else ''}...")

        result = self.call_llm_json(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_message=f"请审查以下 Manim 代码:\n\n```python\n{code}\n```",
            response_model=ReviewResult,
        )

        if result.is_valid:
            self._log("✓ 代码审查通过", style="bold green")
        else:
            self._log(f"✗ 审查未通过: {result.feedback[:100]}...", style="bold red")

        return result
