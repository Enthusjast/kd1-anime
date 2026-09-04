"""受限、可重复的数值数学核验。

符号编译器无法解析所有自然语言中的函数表达式时，本模块使用固定种子
在安全 AST 上进行多点采样。它不会执行 ``eval``，采样通过只读节点递归
解释器完成；采样通过只表示“未发现反例”，不冒充形式证明。
"""

from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Literal

VerificationStatus = Literal["proved", "sampled", "counterexample", "unknown"]


@dataclass(frozen=True, slots=True)
class MathVerification:
    """一次表达式采样的可审计结果。"""

    status: VerificationStatus
    seed: int
    attempted_samples: int = 0
    valid_samples: int = 0
    counterexample: dict[str, float] | None = None
    difference: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SAFE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_SAFE_UNARYOPS = (ast.UAdd, ast.USub)


def _normalise(expression: str) -> str:
    value = str(expression or "")
    value = re.sub(r"\\(?:left|right|cdot|times|quad|,|;)", "", value)
    value = value.replace("²", "^2").replace("³", "^3").replace("−", "-")
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    value = re.sub(r"\s+", "", value)
    # 支持常见的 2x、x(y+1) 简写；函数调用并不在允许的 AST 中，
    # 因此 ``sin(x)`` 不会被误当作可执行函数。
    value = re.sub(r"(?<=[0-9A-Za-z_)])(?=[A-Za-z_(])", "*", value)
    return value


def _parse(expression: str) -> tuple[ast.AST, tuple[str, ...]] | None:
    if re.search(r"\b[A-Za-z_]\w*\s*\(", str(expression or "")):
        return None
    value = _normalise(expression)
    if not value or len(value) > 1_000:
        return None
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        return None
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return None
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _SAFE_BINOPS):
                return None
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _SAFE_UNARYOPS):
                return None
        elif isinstance(node, (ast.operator, ast.unaryop, ast.Expression, ast.Load)):
            continue
        else:
            return None
    return tree.body, tuple(sorted(names))


def _evaluate(node: ast.AST, values: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError("missing variable")
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _SAFE_UNARYOPS):
        value = _evaluate(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _SAFE_BINOPS):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if abs(right) < 1e-9:
                raise ZeroDivisionError
            return left / right
        if abs(right) > 12 or abs(left) > 1_000_000:
            raise ValueError("power outside safe range")
        return left**right
    raise ValueError("unsupported AST node")


def verify_expression_samples(
    left: str,
    right: str,
    *,
    samples: int = 8,
    seed: int = 20260904,
    tolerance: float = 1e-7,
) -> MathVerification:
    """对两个受限算术表达式执行固定种子的多点比较。"""

    left_parsed = _parse(left)
    right_parsed = _parse(right)
    if left_parsed is None or right_parsed is None:
        return MathVerification(status="unknown", seed=seed, reason="表达式超出受限采样语法")
    left_tree, left_names = left_parsed
    right_tree, right_names = right_parsed
    names = tuple(sorted(set(left_names) | set(right_names)))
    rng = random.Random(seed)
    attempted = 0
    valid = 0
    for _ in range(max(1, samples)):
        attempted += 1
        values = {name: round(rng.uniform(-3.0, 3.0), 6) for name in names}
        try:
            left_value = _evaluate(left_tree, values)
            right_value = _evaluate(right_tree, values)
        except (ArithmeticError, ValueError, OverflowError):
            continue
        try:
            finite = math.isfinite(left_value) and math.isfinite(right_value)
        except TypeError:
            continue
        if not finite:
            continue
        valid += 1
        difference = abs(left_value - right_value)
        if difference > tolerance * max(1.0, abs(left_value), abs(right_value)):
            return MathVerification(
                status="counterexample",
                seed=seed,
                attempted_samples=attempted,
                valid_samples=valid,
                counterexample=values,
                difference=difference,
                reason="采样点上的两侧数值不一致",
            )
    if valid == 0:
        return MathVerification(
            status="unknown",
            seed=seed,
            attempted_samples=attempted,
            reason="所有采样点都落在未定义域或非有限值",
        )
    return MathVerification(
        status="sampled",
        seed=seed,
        attempted_samples=attempted,
        valid_samples=valid,
        reason="所有有效采样点未发现反例；这不是形式证明",
    )


def verify_numeric_matrix_product(
    left: str,
    right: str,
    expected: str,
    *,
    tolerance: float = 1e-9,
) -> MathVerification:
    """验证三个纯数字矩阵的乘法和维度，不执行用户表达式。"""

    def parse_matrix(value: str) -> list[list[float]] | None:
        try:
            parsed = ast.literal_eval(_normalise(value))
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, (list, tuple)) or not parsed:
            return None
        rows: list[list[float]] = []
        width: int | None = None
        for row in parsed:
            if not isinstance(row, (list, tuple)) or not row:
                return None
            try:
                converted = [float(item) for item in row]
            except (TypeError, ValueError, OverflowError):
                return None
            if any(not math.isfinite(item) for item in converted):
                return None
            width = width or len(converted)
            if len(converted) != width:
                return None
            rows.append(converted)
        return rows

    left_matrix = parse_matrix(left)
    right_matrix = parse_matrix(right)
    expected_matrix = parse_matrix(expected)
    if left_matrix is None or right_matrix is None or expected_matrix is None:
        return MathVerification(status="unknown", seed=0, reason="矩阵不是纯数字二维字面量")
    if len(left_matrix[0]) != len(right_matrix):
        return MathVerification(
            status="counterexample",
            seed=0,
            reason="矩阵乘法维度不匹配",
        )
    product = [
        [
            sum(
                left_matrix[row][inner] * right_matrix[inner][column]
                for inner in range(len(right_matrix))
            )
            for column in range(len(right_matrix[0]))
        ]
        for row in range(len(left_matrix))
    ]
    if len(product) != len(expected_matrix) or len(product[0]) != len(expected_matrix[0]):
        return MathVerification(
            status="counterexample",
            seed=0,
            reason="矩阵乘积结果维度不匹配",
        )
    for row, values in enumerate(product):
        for column, value in enumerate(values):
            if abs(value - expected_matrix[row][column]) > tolerance:
                return MathVerification(
                    status="counterexample",
                    seed=0,
                    counterexample={"row": float(row), "column": float(column)},
                    difference=abs(value - expected_matrix[row][column]),
                    reason="矩阵乘积存在错误元素",
                )
    return MathVerification(status="proved", seed=0, reason="纯数字矩阵乘法逐项相等")


__all__ = [
    "MathVerification",
    "VerificationStatus",
    "verify_expression_samples",
    "verify_numeric_matrix_product",
]
