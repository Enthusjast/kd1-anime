"""
评估模块 - 提供多维度的动画质量评估功能

- 代码质量评估 (AST 分析、复杂度、可维护性)
- 视觉效果评估 (LLM 分析渲染截图)
- 生成效率评估 (渲染时间、成功率)
"""

from .code_eval import CodeEvaluator
from .evaluator import Evaluator
from .metrics import EvalMetric, EvalResult, QualityScore
from .visual_eval import VisualEvaluator

__all__ = [
    "CodeEvaluator",
    "EvalMetric",
    "EvalResult",
    "Evaluator",
    "QualityScore",
    "VisualEvaluator",
]
