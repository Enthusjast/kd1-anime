"""
代码质量评估器

使用确定性 AST 分析评估 Manim 代码质量。
"""

import ast
import re
from dataclasses import dataclass
from typing import Any, ClassVar

from kd1_anime.agents.validator import (
    ALLOWED_IMPORT_ROOTS,
    BANNED_ATTRIBUTE_NAMES,
    BANNED_CALLS,
    validate_manim_code,
)

from .metrics import EvalMetric, QualityScore


@dataclass
class CodeAnalysisResult:
    """代码分析结果"""

    syntax_valid: bool
    syntax_errors: list[str]
    validation_errors: list[str]
    import_count: int
    class_count: int
    function_count: int
    line_count: int
    max_function_lines: int
    complexity_score: float
    security_issues: list[str]
    style_issues: list[str]


class CodeEvaluator:
    """代码质量评估器

    提供基于 AST 的确定性代码分析功能。
    """

    # 危险函数调用
    DANGEROUS_CALLS: ClassVar[set[str]] = set(BANNED_CALLS)

    def analyze_code(self, code: str) -> CodeAnalysisResult:
        """分析代码质量

        Args:
            code: 要分析的 Python 代码

        Returns:
            CodeAnalysisResult: 分析结果
        """
        result = CodeAnalysisResult(
            syntax_valid=True,
            syntax_errors=[],
            validation_errors=[],
            import_count=0,
            class_count=0,
            function_count=0,
            line_count=0,
            max_function_lines=0,
            complexity_score=0.0,
            security_issues=[],
            style_issues=[],
        )

        # 基本统计
        lines = code.split("\n")
        result.line_count = len(lines)

        # 语法检查
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.syntax_valid = False
            result.syntax_errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return result

        # AST 分析
        self._analyze_ast(tree, result)

        # 安全检查
        self._check_security(tree, result)

        # 风格检查
        self._check_style(code, result)

        # 计算复杂度
        result.complexity_score = self._calculate_complexity(tree)

        # 评估结果必须和真正的提交前安全校验保持一致；仅凭 AST 统计
        # 可能把缺少 Scene、非法 renderer API 或危险能力的代码评为高分。
        validation = validate_manim_code(code)
        result.validation_errors = validation.errors

        return result

    def _analyze_ast(self, tree: ast.AST, result: CodeAnalysisResult):
        """分析 AST 统计信息"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                result.import_count += len(node.names)
            elif isinstance(node, ast.ClassDef):
                result.class_count += 1
            elif isinstance(node, ast.FunctionDef):
                result.function_count += 1
                # 计算函数行数
                if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                    func_lines = node.end_lineno - node.lineno + 1
                    result.max_function_lines = max(result.max_function_lines, func_lines)

    def _check_security(self, tree: ast.AST, result: CodeAnalysisResult):
        """检查代码安全性"""
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            # 检查导入
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in ALLOWED_IMPORT_ROOTS:
                        result.security_issues.append(
                            f"Line {node.lineno}: Banned module import '{alias.name}'"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in ALLOWED_IMPORT_ROOTS:
                        result.security_issues.append(
                            f"Line {node.lineno}: Banned module import from '{node.module}'"
                        )

            # 检查危险函数调用
            elif isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name in self.DANGEROUS_CALLS:
                    result.security_issues.append(
                        f"Line {node.lineno}: Dangerous function call '{func_name}'"
                    )

            elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTE_NAMES:
                result.security_issues.append(
                    f"Line {node.lineno}: Dangerous attribute reference '{node.attr}'"
                )

            elif (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in self.DANGEROUS_CALLS
                and not (isinstance(parents.get(node), ast.Call) and parents[node].func is node)
            ):
                result.security_issues.append(
                    f"Line {node.lineno}: Dangerous capability reference '{node.id}'"
                )

    def _check_style(self, code: str, result: CodeAnalysisResult):
        """检查代码风格"""
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # 行长度检查
            if len(line) > 120:
                result.style_issues.append(f"Line {i}: Line too long ({len(line)} > 120 chars)")

            # 检查尾随空格
            if line.rstrip() != line:
                result.style_issues.append(f"Line {i}: Trailing whitespace")

            # 检查混合制表符和空格
            if "\t" in line and " " in line[: len(line) - len(line.lstrip())]:
                result.style_issues.append(f"Line {i}: Mixed tabs and spaces")

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算代码复杂度 (简化版圈复杂度)"""
        complexity = 1  # 基础复杂度

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                complexity += 1

        return complexity

    def evaluate(self, code: str) -> list[QualityScore]:
        """评估代码质量并返回评分

        Args:
            code: 要评估的 Python 代码

        Returns:
            List[QualityScore]: 各维度评分列表
        """
        analysis = self.analyze_code(code)
        scores = []

        # 语法正确性评分
        syntax_score = 5 if analysis.syntax_valid and not analysis.validation_errors else 1
        syntax_justification = (
            "Code passed syntax and deterministic validation"
            if analysis.syntax_valid and not analysis.validation_errors
            else "; ".join([*analysis.syntax_errors, *analysis.validation_errors])
        )
        scores.append(
            QualityScore(
                metric=EvalMetric.CODE_SYNTAX,
                score=syntax_score,
                justification=syntax_justification,
                details={
                    "syntax_errors": analysis.syntax_errors,
                    "validation_errors": analysis.validation_errors,
                },
            )
        )

        # 安全性评分
        security_score = max(1, 5 - len(analysis.security_issues))
        scores.append(
            QualityScore(
                metric=EvalMetric.CODE_SECURITY,
                score=security_score,
                justification=f"Found {len(analysis.security_issues)} security issues",
                details={"issues": analysis.security_issues},
            )
        )

        # 复杂度评分
        if analysis.complexity_score <= 5:
            complexity_score = 5
        elif analysis.complexity_score <= 10:
            complexity_score = 4
        elif analysis.complexity_score <= 20:
            complexity_score = 3
        elif analysis.complexity_score <= 30:
            complexity_score = 2
        else:
            complexity_score = 1

        scores.append(
            QualityScore(
                metric=EvalMetric.CODE_COMPLEXITY,
                score=complexity_score,
                justification=f"Complexity score: {analysis.complexity_score:.1f}",
                details={
                    "complexity_score": analysis.complexity_score,
                    "line_count": analysis.line_count,
                    "function_count": analysis.function_count,
                    "max_function_lines": analysis.max_function_lines,
                },
            )
        )

        # 代码风格评分
        style_score = max(1, 5 - len(analysis.style_issues) // 3)
        scores.append(
            QualityScore(
                metric=EvalMetric.CODE_STYLE,
                score=style_score,
                justification=f"Found {len(analysis.style_issues)} style issues",
                details={"issues": analysis.style_issues[:10]},  # 只保留前10个
            )
        )

        return scores

    def get_scene_complexity(self, code: str) -> dict[str, Any]:
        """评估场景复杂度

        Args:
            code: Manim 场景代码

        Returns:
            复杂度评估结果
        """
        analysis = self.analyze_code(code)

        # 分析 Manim 特定复杂度
        manim_objects = len(
            re.findall(r"\b(Tex|MathTex|Circle|Square|Line|Arrow|Graph|Axes|NumberPlane)\b", code)
        )
        animations = len(
            re.findall(
                r"\b(Create|FadeIn|FadeOut|Transform|ReplacementTransform|AnimationGroup|Succession)\b",
                code,
            )
        )

        complexity_level = "low"
        if analysis.complexity_score > 20 or manim_objects > 10:
            complexity_level = "very_high"
        elif analysis.complexity_score > 15 or manim_objects > 7:
            complexity_level = "high"
        elif analysis.complexity_score > 10 or manim_objects > 4:
            complexity_level = "medium"

        return {
            "complexity_level": complexity_level,
            "complexity_score": analysis.complexity_score,
            "factors": {
                "object_count": manim_objects,
                "animation_count": animations,
                "line_count": analysis.line_count,
                "function_count": analysis.function_count,
            },
            "estimated_render_time": "slow"
            if complexity_level in ["high", "very_high"]
            else "medium"
            if complexity_level == "medium"
            else "fast",
        }
