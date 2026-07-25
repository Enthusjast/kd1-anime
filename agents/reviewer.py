"""
Reviewer Agent
负责审查 Coder 生成的 Manim 代码

审查清单基于 adithya-s-k/manim_skill 的常见陷阱和最佳实践
"""

from pydantic import BaseModel

from agents.base import BaseAgent

REVIEWER_SYSTEM_PROMPT = r"""你是 Manim 代码审查专家.

## 审查清单
(略, 同原 28 项检查)

## 问题分级 — 关键

对每个发现的问题, 判断严重程度:
- **minor**: 局部修改即可修复 (如改个类名、加个参数、删一行). 给出精确的查找替换指令.
- **major**: 需要重写大段逻辑、调整整体结构、或问题太多无法逐条修. 给出详细反馈交给 Coder 重写.

## 输出 JSON
{
  "is_valid": true,
  "severity": "major",
  "feedback": "问题描述 (major 时必须填)",
  "fixes": []
}

fixes 每条: {"find": "原代码片段(唯一匹配)", "replace": "替换后代码", "reason": "原因"}
find 必须是原代码中唯一切实的片段 (建议包含上下文 2-3 行), 否则替换会失败.
如果一个修改需要在多处执行 (如全局替换类名), 给多个 fix 条目.
"""


class FixSuggestion(BaseModel):
    """单条查找替换"""
    find: str
    replace: str
    reason: str = ""


class ReviewResult(BaseModel):
    """审查结果"""
    is_valid: bool
    severity: str = "minor"  # "minor" → 自动修复 | "major" → 交给 Coder
    feedback: str = ""
    fixes: list[FixSuggestion] = []


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
