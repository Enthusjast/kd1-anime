"""分镜计划的轻量确定性编译器。

Planner 负责创意，Plan Reviewer 负责语义判断；本模块只做不需要 LLM 的
机械检查。它不会假装拥有完整计算机代数系统，只在能够明确判断时报告
错误，无法判断的表达式留给计划审查模型。
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.agents.planner import ContinuityBible, SceneOutline, ScenePlan


class PlanCompilerIssue(BaseModel):
    """确定性计划错误，字段与计划审查输出保持一致。"""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "math",
        "geometry",
        "feasibility",
        "timing",
        "continuity",
        "contract",
        "renderer",
        "style",
    ]
    severity: Literal["minor", "major"] = "major"
    scene_ids: list[int] = Field(default_factory=list, max_length=20)
    field: str = Field(default="", max_length=200)
    message: str = Field(min_length=1, max_length=5_000)
    fix_instruction: str = Field(min_length=1, max_length=5_000)


class PlanCompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    issues: list[PlanCompilerIssue] = Field(default_factory=list, max_length=200)


_TOKEN_RE = re.compile(r"\*\*|[()+\-*/^]|(?:\d+(?:\.\d+)?)|[A-Za-z]+")


class _PolynomialParser:
    """仅支持常见初等展开的微型多项式解析器。"""

    def __init__(self, expression: str) -> None:
        self.tokens = self._tokens(expression)
        self.index = 0

    @staticmethod
    def _tokens(expression: str) -> list[str]:
        normalized = expression
        normalized = re.sub(r"\\(?:left|right|cdot|times|,|;|!|quad|,)", "", normalized)
        normalized = normalized.replace("{", "(").replace("}", ")")
        normalized = normalized.replace("²", "^2").replace("³", "^3")
        normalized = normalized.replace("−", "-").replace("·", "*")
        compact = re.sub(r"\s+", "", normalized)
        tokens: list[str] = []
        position = 0
        for match in _TOKEN_RE.finditer(compact):
            if match.start() != position:
                raise ValueError("unsupported token")
            token = match.group(0)
            if token.isalpha() and len(token) > 1:
                # 常见的无显式乘号写法 2ab；只接受单字母变量，
                # 多字母单词不应被当成数学式误判。
                if token.lower() in {"pi"}:
                    tokens.append(token.lower())
                else:
                    tokens.extend(token)
            else:
                tokens.append(token)
            position = match.end()
        if position != len(compact):
            raise ValueError("unsupported token")
        return tokens

    @staticmethod
    def _add(left: dict[tuple[str, ...], Fraction], right: dict[tuple[str, ...], Fraction], sign=1):
        result = dict(left)
        for monomial, coefficient in right.items():
            result[monomial] = result.get(monomial, Fraction(0)) + sign * coefficient
            if result[monomial] == 0:
                del result[monomial]
        return result

    @staticmethod
    def _multiply(
        left: dict[tuple[str, ...], Fraction],
        right: dict[tuple[str, ...], Fraction],
    ):
        result: dict[tuple[str, ...], Fraction] = {}
        for left_monomial, left_coefficient in left.items():
            for right_monomial, right_coefficient in right.items():
                monomial = tuple(sorted((*left_monomial, *right_monomial)))
                result[monomial] = result.get(monomial, Fraction(0)) + (
                    left_coefficient * right_coefficient
                )
        return {key: value for key, value in result.items() if value}

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, token: str | None = None) -> str:
        current = self.peek()
        if current is None or (token is not None and current != token):
            raise ValueError("unexpected token")
        self.index += 1
        return current

    def parse(self):
        result = self.parse_sum()
        if self.peek() is not None:
            raise ValueError("trailing token")
        return result

    def parse_sum(self):
        result = self.parse_product()
        while self.peek() in {"+", "-"}:
            sign = self.take()
            result = self._add(result, self.parse_product(), -1 if sign == "-" else 1)
        return result

    def parse_product(self):
        result = self.parse_power()
        while True:
            if self.peek() == "*":
                self.take("*")
                result = self._multiply(result, self.parse_power())
                continue
            if self.peek() in {"(", "pi"} or (
                self.peek() is not None and (self.peek().isalpha() or self.peek()[0].isdigit())
            ):
                result = self._multiply(result, self.parse_power())
                continue
            break
        return result

    def parse_power(self):
        sign = 1
        if self.peek() in {"+", "-"}:
            sign = -1 if self.take() == "-" else 1
        atom = self.parse_atom()
        if self.peek() in {"^", "**"}:
            self.take()
            exponent = int(self.take())
            if exponent < 0 or exponent > 12:
                raise ValueError("unsupported exponent")
            result: dict[tuple[str, ...], Fraction] = {(): Fraction(1)}
            for _ in range(exponent):
                result = self._multiply(result, atom)
        else:
            result = atom
        if sign < 0:
            result = {monomial: -coefficient for monomial, coefficient in result.items()}
        return result

    def parse_atom(self):
        token = self.peek()
        if token == "(":
            self.take("(")
            result = self.parse_sum()
            self.take(")")
            return result
        if token == "pi":
            self.take()
            # 只用于比较含 pi 的相同表达式；用一个符号比近似浮点更安全。
            return {("pi",): Fraction(1)}
        if token is not None and token.isalpha():
            self.take()
            return {(token,): Fraction(1)}
        if token is not None and re.fullmatch(r"\d+(?:\.\d+)?", token):
            self.take()
            return {(): Fraction(token)}
        raise ValueError("expected atom")


def _normalise_expression(expression: str) -> str:
    value = str(expression or "")
    value = re.sub(r"\\(?:left|right|cdot|times|quad|,|;)", "", value)
    value = value.replace("²", "^2").replace("³", "^3").replace("−", "-")
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"\s+", "", value)
    return value


def _polynomial(expression: str) -> dict[tuple[str, ...], Fraction] | None:
    try:
        return _PolynomialParser(_normalise_expression(expression)).parse()
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_numeric(expression: str) -> float | None:
    value = _normalise_expression(expression).replace("^", "**")
    if not value or not re.fullmatch(r"[0-9eE.+\-*/() ]+", value):
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = evaluate(node.operand)
            return result if isinstance(node.op, ast.UAdd) else -result
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if abs(right) > 12 or abs(left) > 1_000_000:
                raise ValueError("numeric expression is too large")
            return left**right
        raise ValueError("unsupported numeric expression")

    try:
        tree = ast.parse(value, mode="eval")
        return evaluate(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def expressions_are_equivalent(left: str, right: str) -> bool | None:
    """返回 True/False；无法安全解析时返回 None。"""

    left_value = _polynomial(left)
    right_value = _polynomial(right)
    if left_value is not None and right_value is not None:
        return left_value == right_value
    left_number = _safe_numeric(left)
    right_number = _safe_numeric(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= 1e-9
    return None


def _simple_equations(text: str) -> list[tuple[str, str]]:
    """从 computation 中提取足够短的符号等式，忽略变量赋值。"""

    pairs: list[tuple[str, str]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])([A-Za-z0-9_()+\-*/^²³.]{1,80})\s*(?:=|→|⟶)\s*"
        r"([A-Za-z0-9_()+\-*/^²³.]{1,80})(?![A-Za-z0-9_])"
    )
    for match in pattern.finditer(str(text or "")):
        left, right = match.groups()
        if (
            any(char in left + right for char in "+-*/^²³")
            and re.search(r"[A-Za-z]", left + right)
            and not re.fullmatch(r"[A-Za-z]", left)
            and set(re.findall(r"[A-Za-z]", left)) == set(re.findall(r"[A-Za-z]", right))
        ):
            pairs.append((left, right))
    return pairs


class PlanCompiler:
    """对整部动画计划执行一次确定性编译。"""

    def compile_scene(
        self,
        plan: ScenePlan,
        bible: ContinuityBible | None = None,
    ) -> list[PlanCompilerIssue]:
        """只检查一个场景，不检查整片编号和相邻边界。"""

        return self._compile_scene(plan, bible)

    def compile(
        self,
        outlines: list[SceneOutline],
        plans: list[ScenePlan],
        bible: ContinuityBible | None = None,
    ) -> PlanCompileResult:
        issues: list[PlanCompilerIssue] = []
        ordered_plans = sorted(plans, key=lambda item: item.scene_id)
        outline_ids = [item.scene_id for item in sorted(outlines, key=lambda item: item.scene_id)]
        plan_ids = [item.scene_id for item in ordered_plans]
        if outline_ids != list(range(1, len(outlines) + 1)):
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="scene_id",
                    message="场景概要 ID 必须从 1 开始连续编号。",
                    fix_instruction="按叙事顺序重新编号为 1,2,...,N。",
                )
            )
        if plan_ids != list(range(1, len(ordered_plans) + 1)):
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="scene_id",
                    message="详细分镜 ID 与场景数量/顺序不一致。",
                    fix_instruction="为每个概要提供一个同 ID 的详细分镜，并按 1..N 排序。",
                    scene_ids=plan_ids,
                )
            )
        if len({item.scene_id for item in plans}) != len(plans):
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="scene_id",
                    message="详细分镜包含重复 scene_id。",
                    fix_instruction="删除重复分镜，不要让同一个 Scene ID 对应多个计划。",
                    scene_ids=plan_ids,
                )
            )

        previous: ScenePlan | None = None
        for plan in ordered_plans:
            issues.extend(self._compile_scene(plan, bible))
            if previous is not None:
                issues.extend(self._compile_boundary(previous, plan))
            previous = plan
        return PlanCompileResult(is_valid=not issues, issues=issues)

    def _compile_scene(
        self,
        plan: ScenePlan,
        bible: ContinuityBible | None,
    ) -> list[PlanCompilerIssue]:
        issues: list[PlanCompilerIssue] = []
        scene_ids = [plan.scene_id]
        if plan.timeline:
            events = sorted(plan.timeline, key=lambda item: item.start_seconds)
            if events[0].start_seconds > 0.05:
                issues.append(
                    PlanCompilerIssue(
                        category="timing",
                        field="timeline",
                        scene_ids=scene_ids,
                        message="时间线没有从场景开始覆盖。",
                        fix_instruction="将第一个事件 start_seconds 设为 0，并覆盖开场状态。",
                    )
                )
            cursor = events[0].end_seconds
            for event in events[1:]:
                if event.start_seconds > cursor + 0.05:
                    issues.append(
                        PlanCompilerIssue(
                            category="timing",
                            field="timeline",
                            scene_ids=scene_ids,
                            message=f"时间线在 {cursor:.2f}s 到 {event.start_seconds:.2f}s 之间存在空档。",
                            fix_instruction="补充该时间段的停顿或视觉事件，保证教学过程连续。",
                        )
                    )
                cursor = max(cursor, event.end_seconds)
            if cursor < plan.duration_seconds - 0.05:
                issues.append(
                    PlanCompilerIssue(
                        category="timing",
                        field="timeline",
                        scene_ids=scene_ids,
                        message=f"时间线只覆盖到 {cursor:.2f}s，短于场景时长 {plan.duration_seconds:.2f}s。",
                        fix_instruction="补充结论定格/吸收停顿，使最后一个事件覆盖到场景结束。",
                    )
                )
            event_ids = [event.event_id for event in plan.timeline]
            if len(event_ids) != len(set(event_ids)):
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="timeline.event_id",
                        scene_ids=scene_ids,
                        message="timeline.event_id 必须唯一。",
                        fix_instruction="为每个时间线事件分配稳定且唯一的 event_id。",
                    )
                )
            claim_ids = {claim.claim_id for claim in plan.math_claims}
            missing_claims = {
                claim_id
                for event in plan.timeline
                for claim_id in event.math_claim_ids
                if claim_id not in claim_ids
            }
            if missing_claims:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field="timeline.math_claim_ids",
                        scene_ids=scene_ids,
                        message="时间线引用了不存在的数学断言: "
                        + ", ".join(sorted(missing_claims)),
                        fix_instruction="在 math_claims 中补充断言，或删除错误的引用。",
                    )
                )

        for claim in plan.math_claims:
            left = claim.expression_before
            right = claim.expression_after
            if not left or not right:
                equality = re.split(r"=|≡|→|⟶", claim.statement, maxsplit=1)
                if len(equality) == 2:
                    left, right = equality
            if not left or not right:
                continue
            equivalent = expressions_are_equivalent(
                left,
                right,
            )
            if equivalent is False and claim.relation in {"equivalent", "equals", "area"}:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field=f"math_claims[{claim.claim_id}]",
                        scene_ids=scene_ids,
                        message=(
                            f"数学断言 {claim.claim_id} 的前后表达式不等价：{left} ≠ {right}。"
                        ),
                        fix_instruction="修正表达式或将 relation 改为真实的非等价关系，并说明推导依据。",
                    )
                )

        for index, (left, right) in enumerate(_simple_equations(plan.computation), start=1):
            if expressions_are_equivalent(left, right) is False:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field=f"computation.equation_{index}",
                        scene_ids=scene_ids,
                        message=f"computation 中的等式不等价：{left} ≠ {right}。",
                        fix_instruction="修正等式，或把变量赋值与需要证明的等式分开书写。",
                    )
                )

        for geometry in plan.geometry_specs:
            if not geometry.vertices:
                continue
            if len(geometry.vertices) < 3:
                issues.append(
                    PlanCompilerIssue(
                        category="geometry",
                        field=f"geometry_specs[{geometry.geometry_id}].vertices",
                        scene_ids=scene_ids,
                        message="多边形至少需要三个顶点。",
                        fix_instruction="补充完整顶点，或不要把该对象声明为可核验多边形。",
                    )
                )
                continue
            if any(len(point) < 2 for point in geometry.vertices):
                issues.append(
                    PlanCompilerIssue(
                        category="geometry",
                        field=f"geometry_specs[{geometry.geometry_id}].vertices",
                        scene_ids=scene_ids,
                        message="几何顶点必须至少包含 x、y 坐标。",
                        fix_instruction="为每个顶点提供二维坐标。",
                    )
                )
                continue
            area = abs(
                sum(
                    point[0] * next_point[1] - next_point[0] * point[1]
                    for point, next_point in zip(
                        geometry.vertices,
                        [*geometry.vertices[1:], *geometry.vertices[:1]],
                        strict=True,
                    )
                )
                / 2
            )
            if geometry.declared_area is not None and abs(area - geometry.declared_area) > 1e-6:
                issues.append(
                    PlanCompilerIssue(
                        category="geometry",
                        field=f"geometry_specs[{geometry.geometry_id}].declared_area",
                        scene_ids=scene_ids,
                        message=f"顶点鞋带公式面积为 {area:g}，但计划声明为 {geometry.declared_area:g}。",
                        fix_instruction="修正顶点或 declared_area，使几何面积一致。",
                    )
                )
            if geometry.target_area is not None and abs(area - geometry.target_area) > 1e-6:
                issues.append(
                    PlanCompilerIssue(
                        category="geometry",
                        field=f"geometry_specs[{geometry.geometry_id}].target_area",
                        scene_ids=scene_ids,
                        message=f"几何面积 {area:g} 与目标面积 {geometry.target_area:g} 不一致。",
                        fix_instruction="修正目标覆盖关系；不能用视觉上的接近代替面积相等。",
                    )
                )
            if any(abs(point[0]) > 7.2 or abs(point[1]) > 4.2 for point in geometry.vertices):
                issues.append(
                    PlanCompilerIssue(
                        category="feasibility",
                        field=f"geometry_specs[{geometry.geometry_id}].vertices",
                        scene_ids=scene_ids,
                        message="几何顶点超出默认 16:9 安全画布范围。",
                        fix_instruction="将顶点调整到约 x∈[-7,7]、y∈[-4,4] 的安全区，或明确镜头移动。",
                    )
                )

        declared_ids = {
            item.element_id
            for item in [*plan.inherited_elements, *plan.new_elements]
            if item.element_id not in {removed.element_id for removed in plan.elements_to_remove}
        }
        handoff_ids = {item.element_id for item in plan.handoff}
        if len(handoff_ids) != len(plan.handoff):
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="handoff.element_id",
                    scene_ids=scene_ids,
                    message="handoff 中的 element_id 必须唯一。",
                    fix_instruction="合并重复声明，并为每个元素保留一个生命周期动作。",
                )
            )
        undeclared_handoff = handoff_ids - declared_ids
        if undeclared_handoff:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="handoff",
                    scene_ids=scene_ids,
                    message="handoff 引用了未声明的元素: " + ", ".join(sorted(undeclared_handoff)),
                    fix_instruction="让 handoff 与 inherited_elements/new_elements 使用相同的 element_id。",
                )
            )
        inherited_ids = {item.element_id for item in plan.inherited_elements}
        new_ids = {item.element_id for item in plan.new_elements}
        removed_ids = {item.element_id for item in plan.elements_to_remove}
        for item in plan.handoff:
            expected_actions = (
                {"remove"}
                if item.element_id in removed_ids
                else {"inherit", "keep"}
                if item.element_id in inherited_ids
                else {"create", "keep"}
                if item.element_id in new_ids
                else set()
            )
            if expected_actions and item.action not in expected_actions:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field=f"handoff[{item.element_id}].action",
                        scene_ids=scene_ids,
                        message=(
                            f"元素 {item.element_id} 的 handoff 动作 {item.action} "
                            f"与声明不一致，应为 {sorted(expected_actions)}。"
                        ),
                        fix_instruction="按 inherited/new/elements_to_remove 的生命周期修正 handoff.action。",
                    )
                )
        if bible is not None and plan.global_visual_state != bible.global_visual_state:
            issues.append(
                PlanCompilerIssue(
                    category="style",
                    field="global_visual_state",
                    scene_ids=scene_ids,
                    message="场景全局视觉配置与连续性圣经不一致。",
                    fix_instruction="完整复制连续性圣经的颜色、字体、字号和线宽配置。",
                )
            )
        return issues

    @staticmethod
    def _compile_boundary(previous: ScenePlan, current: ScenePlan) -> list[PlanCompilerIssue]:
        previous_available = {
            item.element_id
            for item in [*previous.inherited_elements, *previous.new_elements]
            if item.element_id
            not in {removed.element_id for removed in previous.elements_to_remove}
        }
        current_inherited = {item.element_id for item in current.inherited_elements}
        if not previous_available or not current_inherited:
            return []
        missing = current_inherited - previous_available
        if not missing:
            return []
        return [
            PlanCompilerIssue(
                category="continuity",
                field="inherited_elements",
                scene_ids=[previous.scene_id, current.scene_id],
                message=(
                    f"Scene {current.scene_id} 继承了 Scene {previous.scene_id} 未交接的元素: "
                    + ", ".join(sorted(missing))
                ),
                fix_instruction="只继承上一场景 closing_state 中真实存在并在导出合同中保留的元素。",
            )
        ]
