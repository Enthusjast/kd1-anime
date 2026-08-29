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

from kd1_anime.agents.planner import (
    ContinuityBible,
    LessonSpec,
    SceneOutline,
    ScenePlan,
    TeachingGraph,
)
from kd1_anime.rendering import effective_transition_duration


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


def _is_exit_timeline_action(action: str) -> bool:
    text = str(action or "").lower()
    return any(
        token in text for token in ("淡出", "消失", "清空", "收束", "结束", "fade out", "fadeout")
    )


def _is_deferred_transition_exit(text: str) -> bool:
    """判断转场中的退出是否发生在当前场景边界之后。"""

    normalized = str(text or "").lower()
    return (_is_exit_timeline_action(normalized) or "self.clear" in normalized) and any(
        marker in normalized
        for marker in (
            "下一场景",
            "下一个场景",
            "后续场景",
            "进入下一",
            "场景切换",
            "next scene",
            "following scene",
        )
    )


def normalize_scene_timeline_contract(
    plan: ScenePlan,
) -> tuple[ScenePlan, tuple[str, ...]]:
    """吸收末尾未覆盖的时间到结论定格，避免模型遗漏长停顿。

    Detail 常能正确写出 ``key_moments`` 的长时间定格，却把 timeline
    只写到最后一个动画动作。若最后一个事件是淡出，应把淡出推迟到场景
    末尾，并延长前一个内容事件；否则直接延长最后一个内容事件。这个
    归一化只改变停顿分配，不改变对象、数学断言或动画顺序。
    """

    if not plan.timeline:
        return plan, ()
    ordered = sorted(
        enumerate(plan.timeline),
        key=lambda pair: (pair[1].start_seconds, pair[1].event_id, pair[0]),
    )
    last_index, last_event = ordered[-1]
    gap = plan.duration_seconds - last_event.end_seconds
    if gap <= 0.05:
        return plan, ()

    events = list(plan.timeline)
    repairs: list[str] = []
    if _is_exit_timeline_action(last_event.action) and len(ordered) >= 2:
        previous_index, previous_event = ordered[-2]
        if not _is_exit_timeline_action(previous_event.action):
            shifted_start = last_event.start_seconds + gap
            if shifted_start > previous_event.start_seconds + 0.05:
                events[previous_index] = previous_event.model_copy(
                    update={"end_seconds": shifted_start}
                )
                events[last_index] = last_event.model_copy(
                    update={
                        "start_seconds": shifted_start,
                        "end_seconds": plan.duration_seconds,
                    }
                )
                repairs.append(
                    f"将收束事件 {last_event.event_id} 延后至场景末尾，并补足 {gap:.2f}s 结论定格"
                )
            else:
                events[last_index] = last_event.model_copy(
                    update={"end_seconds": plan.duration_seconds}
                )
                repairs.append(f"将时间线最后事件 {last_event.event_id} 延长至场景末尾")
        else:
            events[last_index] = last_event.model_copy(
                update={"end_seconds": plan.duration_seconds}
            )
            repairs.append(f"将时间线最后事件 {last_event.event_id} 延长至场景末尾")
    else:
        events[last_index] = last_event.model_copy(update={"end_seconds": plan.duration_seconds})
        repairs.append(f"将时间线最后事件 {last_event.event_id} 延长至场景末尾")

    return plan.model_copy(update={"timeline": events}), tuple(repairs)


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
        # 将常数 0 规范化为空多项式，确保 ``0`` 与 ``-ab + ab``
        # 在消项后得到同一个表示，而不会被字典中的零系数误判为不等价。
        return {monomial: coefficient for monomial, coefficient in result.items() if coefficient}

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
        lesson_spec: LessonSpec | None = None,
    ) -> list[PlanCompilerIssue]:
        """只检查一个场景，不检查整片编号和相邻边界。"""

        return self._compile_scene(plan, bible, lesson_spec)

    def compile(
        self,
        outlines: list[SceneOutline],
        plans: list[ScenePlan],
        bible: ContinuityBible | None = None,
        lesson_spec: LessonSpec | None = None,
        teaching_graph: TeachingGraph | None = None,
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
        expected_plan_ids = list(range(1, len(outlines) + 1))
        if plan_ids != expected_plan_ids or plan_ids != outline_ids:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="scene_id",
                    message="详细分镜 ID 与场景数量/顺序不一致。",
                    fix_instruction="为每个概要提供一个同 ID 的详细分镜，并按 1..N 排序。",
                    scene_ids=plan_ids or outline_ids,
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
            issues.extend(self._compile_scene(plan, bible, lesson_spec))
            if previous is not None:
                issues.extend(self._compile_boundary(previous, plan))
            previous = plan
        if lesson_spec is not None:
            issues.extend(
                self._compile_lesson_contract(
                    ordered_plans,
                    lesson_spec,
                    teaching_graph=teaching_graph,
                )
            )
        return PlanCompileResult(is_valid=not issues, issues=issues)

    def _compile_scene(
        self,
        plan: ScenePlan,
        bible: ContinuityBible | None,
        lesson_spec: LessonSpec | None = None,
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
            for event in events:
                if event.end_seconds > plan.duration_seconds + 0.05:
                    issues.append(
                        PlanCompilerIssue(
                            category="timing",
                            field="timeline",
                            scene_ids=scene_ids,
                            message=(
                                f"时间线事件 {event.event_id} 结束于 {event.end_seconds:.2f}s，"
                                f"超出场景时长 {plan.duration_seconds:.2f}s。"
                            ),
                            fix_instruction="将事件结束时间调整到场景时长以内，或增加场景总时长。",
                        )
                    )
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
            if (
                claim.relation in {"equivalent", "equals", "inequality"}
                and not claim.domain.strip()
                and not claim.assumptions
                and any(
                    token in (left + right).lower()
                    for token in ("/", "\\frac", "sqrt", "log", "ln", "根号")
                )
            ):
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field=f"math_claims[{claim.claim_id}].domain",
                        scene_ids=scene_ids,
                        message="含除法、根式或对数的等价断言没有声明定义域或前提条件。",
                        fix_instruction="补充 domain 或 assumptions，说明断言成立的条件。",
                    )
                )

        if plan.claim_ids:
            declared_claim_ids = set(plan.claim_ids)
            detail_claim_ids = {claim.claim_id for claim in plan.math_claims}
            timeline_claim_ids = {
                claim_id for event in plan.timeline for claim_id in event.math_claim_ids
            }
            missing_detail_claims = declared_claim_ids - detail_claim_ids
            extra_detail_claims = detail_claim_ids - declared_claim_ids
            if missing_detail_claims:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field="math_claims",
                        scene_ids=scene_ids,
                        message="场景概要声明的断言没有对应的详细数学断言: "
                        + ", ".join(sorted(missing_detail_claims)),
                        fix_instruction="为每个 claim_id 填写可核验的 math_claims，并在时间线中展示其依据。",
                    )
                )
            if extra_detail_claims:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field="math_claims",
                        scene_ids=scene_ids,
                        message="详细分镜增加了概要未声明的数学断言: "
                        + ", ".join(sorted(extra_detail_claims)),
                        fix_instruction="删除额外断言，或先把它纳入全片 LessonSpec 和当前场景 claim_ids。",
                    )
                )
            if not plan.timeline:
                issues.append(
                    PlanCompilerIssue(
                        category="timing",
                        field="timeline",
                        scene_ids=scene_ids,
                        message="场景声明了数学断言，但没有时间线画面证据。",
                        fix_instruction="为每个 claim_id 添加覆盖场景时长的 timeline 事件，并绑定 math_claim_ids。",
                    )
                )
            if plan.timeline:
                missing_timeline_claims = declared_claim_ids - timeline_claim_ids
                if missing_timeline_claims:
                    issues.append(
                        PlanCompilerIssue(
                            category="math",
                            field="timeline.math_claim_ids",
                            scene_ids=scene_ids,
                            message="时间线没有为场景断言提供画面证据: "
                            + ", ".join(sorted(missing_timeline_claims)),
                            fix_instruction="将每个场景 claim_id 绑定到至少一个时间线事件。",
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
            if geometry.shape == "circle":
                # GeometrySpec 没有圆心/半径字段，不能把圆的离散辅助点
                # 当作多边形用鞋带公式计算面积；圆的语义审查交给 LLM。
                continue
            if geometry.shape == "line":
                if len(geometry.vertices) < 2:
                    issues.append(
                        PlanCompilerIssue(
                            category="geometry",
                            field=f"geometry_specs[{geometry.geometry_id}].vertices",
                            scene_ids=scene_ids,
                            message="线段至少需要两个顶点。",
                            fix_instruction="补充线段的起点和终点，或不要声明该几何对象。",
                        )
                    )
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

        inherited_ids = {item.element_id for item in plan.inherited_elements}
        new_ids = {item.element_id for item in plan.new_elements}
        removed_ids = {item.element_id for item in plan.elements_to_remove}
        # removed 元素仍然是本场景的合法声明，因为 handoff 需要用
        # ``action=remove`` 显式描述它的退出。最终导出区是否包含它会由
        # validate_export_contract() 单独拒绝。
        declared_ids = inherited_ids | new_ids | removed_ids
        handoff_ids = {item.element_id for item in plan.handoff}
        timeline_element_ids = {
            element_id for event in plan.timeline for element_id in event.element_ids
        }
        unknown_timeline_elements = timeline_element_ids - declared_ids
        if unknown_timeline_elements:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="timeline.element_ids",
                    scene_ids=scene_ids,
                    message="时间线引用了未声明的元素: "
                    + ", ".join(sorted(unknown_timeline_elements)),
                    fix_instruction="让 timeline.element_ids 与 inherited_elements、new_elements 或 elements_to_remove 对齐。",
                )
            )
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
        missing_required_handoff = {
            item.element_id for item in plan.new_elements if item.required
        } - handoff_ids
        if missing_required_handoff:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="handoff",
                    scene_ids=scene_ids,
                    message=(
                        "required=true 的 new_elements 未列入 handoff: "
                        + ", ".join(sorted(missing_required_handoff))
                    ),
                    fix_instruction="为每个需要交给下一场景的 new_element 增加 handoff 条目，并使用 create 或 keep。",
                )
            )
        invalid_removed = removed_ids - inherited_ids
        if invalid_removed:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="elements_to_remove",
                    scene_ids=scene_ids,
                    message=(
                        "elements_to_remove 只能引用 inherited_elements 中的元素: "
                        + ", ".join(sorted(invalid_removed))
                    ),
                    fix_instruction="只移除本场景真实继承的元素，或先加入 inherited_elements。",
                )
            )
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
        required_boundary_ids = {
            item.element_id
            for item in [*plan.inherited_elements, *plan.new_elements]
            if item.required and item.element_id not in removed_ids
        }
        closing_text = " ".join(plan.closing_state).lower()
        transition_text = plan.transition_out.lower()
        broad_exit_pattern = r"(?:所有|全部|整体|全片).{0,16}(?:淡出|消失|清空|移除)"
        closing_broad_exit = bool(
            re.search(broad_exit_pattern, closing_text) or "self.clear" in closing_text
        )
        transition_broad_exit = bool(
            re.search(broad_exit_pattern, transition_text) or "self.clear" in transition_text
        )
        # “下一场景淡入时，本场景元素淡出”表示先在当前边界保留，
        # 再由下一场景处理退出，不能被误判为 closing_state 的冲突。
        broad_exit = closing_broad_exit or (
            transition_broad_exit and not _is_deferred_transition_exit(transition_text)
        )
        if required_boundary_ids and broad_exit:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="transition_out|closing_state",
                    scene_ids=scene_ids,
                    message=(
                        "场景声明了必须交接的元素，但 transition_out/closing_state 又要求整体退出: "
                        + ", ".join(sorted(required_boundary_ids))
                    ),
                    fix_instruction="保留 required 元素到场景边界并在 handoff 中 keep/create，或将只退出的元素标为 optional/明确列入移除合同。",
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
        if lesson_spec is not None and lesson_spec.claims:
            known_claim_ids = {claim.claim_id for claim in lesson_spec.claims}
            unknown_outline_claims = set(plan.claim_ids) - known_claim_ids
            unknown_detail_claims = {claim.claim_id for claim in plan.math_claims} - known_claim_ids
            unknown_claims = unknown_outline_claims | unknown_detail_claims
            if unknown_claims:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field="claim_ids|math_claims",
                        scene_ids=scene_ids,
                        message="场景引用或新增了教学合同未声明的数学断言: "
                        + ", ".join(sorted(unknown_claims)),
                        fix_instruction="只引用 LessonSpec 中已有的 claim_id；若确需新增断言，先修改全片教学合同。",
                    )
                )
        return issues

    @staticmethod
    def _compile_lesson_contract(
        plans: list[ScenePlan],
        lesson_spec: LessonSpec,
        *,
        teaching_graph: TeachingGraph | None = None,
    ) -> list[PlanCompilerIssue]:
        """检查全片断言覆盖、依赖顺序和最终成片时长。"""

        if not plans:
            return (
                [
                    PlanCompilerIssue(
                        category="contract",
                        field="scene_claims",
                        scene_ids=[1],
                        message="LessonSpec 已存在，但没有可执行的场景计划。",
                        fix_instruction="至少提供一个 ScenePlan，并为其分配教学断言。",
                    )
                ]
                if lesson_spec.claims
                else []
            )
        issues: list[PlanCompilerIssue] = []
        covered = {claim_id for plan in plans for claim_id in plan.claim_ids}
        missing_core = {
            claim.claim_id for claim in lesson_spec.claims if claim.claim_id not in covered
        }
        if missing_core:
            issues.append(
                PlanCompilerIssue(
                    category="math",
                    field="claim_ids",
                    scene_ids=[plan.scene_id for plan in plans],
                    message="全片概要没有覆盖教学合同中的数学断言: "
                    + ", ".join(sorted(missing_core)),
                    fix_instruction="为负责讲解这些断言的场景补充对应 claim_ids，不能只在自然语言中提及。",
                )
            )

        known_claim_ids = set(
            lesson_spec_claim.claim_id for lesson_spec_claim in lesson_spec.claims
        )
        for claim in lesson_spec.claims:
            unknown_prerequisites = set(claim.prerequisite_claim_ids) - known_claim_ids
            if unknown_prerequisites:
                issues.append(
                    PlanCompilerIssue(
                        category="math",
                        field=f"claims[{claim.claim_id}].prerequisite_claim_ids",
                        scene_ids=[plan.scene_id for plan in plans],
                        message=(
                            f"断言 {claim.claim_id} 引用了不存在的前置断言: "
                            + ", ".join(sorted(unknown_prerequisites))
                        ),
                        fix_instruction="删除未知前置断言，或先在 LessonSpec 中声明它。",
                    )
                )

        # Kahn 算法检测教学依赖环；有环时无法得到稳定的讲解顺序。
        indegree = {claim_id: 0 for claim_id in known_claim_ids}
        adjacency: dict[str, set[str]] = {claim_id: set() for claim_id in known_claim_ids}
        for claim in lesson_spec.claims:
            for prerequisite in claim.prerequisite_claim_ids:
                if (
                    prerequisite in known_claim_ids
                    and claim.claim_id not in adjacency[prerequisite]
                ):
                    adjacency[prerequisite].add(claim.claim_id)
                    indegree[claim.claim_id] += 1
        queue = [claim_id for claim_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for dependent in adjacency[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if visited != len(indegree):
            issues.append(
                PlanCompilerIssue(
                    category="math",
                    field="claims.prerequisite_claim_ids",
                    scene_ids=[plan.scene_id for plan in plans],
                    message="LessonSpec 的数学断言依赖存在环，无法安排教学顺序。",
                    fix_instruction="删除循环依赖，建立从前置知识到结论的有向无环顺序。",
                )
            )

        if teaching_graph is not None:
            graph_order = list(teaching_graph.claim_order)
            graph_order_set = set(graph_order)
            if known_claim_ids and not graph_order:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.claim_order",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="LessonSpec 已声明数学断言，但教学图谱没有 claim_order。",
                        fix_instruction="按前置依赖顺序补充所有 LessonSpec.claim_id。",
                    )
                )
            duplicate_graph_nodes = {
                claim_id for claim_id in graph_order if graph_order.count(claim_id) > 1
            }
            if duplicate_graph_nodes:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.claim_order",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="教学图谱 claim_order 包含重复断言: "
                        + ", ".join(sorted(duplicate_graph_nodes)),
                        fix_instruction="每个 claim_id 在 claim_order 中只出现一次。",
                    )
                )
            unknown_graph_nodes = graph_order_set - known_claim_ids
            missing_graph_nodes = known_claim_ids - graph_order_set if graph_order else set()
            if unknown_graph_nodes:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.claim_order",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="教学图谱 claim_order 引用了未声明的断言: "
                        + ", ".join(sorted(unknown_graph_nodes)),
                        fix_instruction="只保留 LessonSpec 中声明的 claim_id。",
                    )
                )
            if missing_graph_nodes:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.claim_order",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="教学图谱 claim_order 缺少 LessonSpec 断言: "
                        + ", ".join(sorted(missing_graph_nodes)),
                        fix_instruction="把所有教学断言按前置依赖顺序放入 claim_order。",
                    )
                )

            plan_by_scene = {plan.scene_id: set(plan.claim_ids) for plan in plans}
            graph_scene_ids = set(teaching_graph.scene_claims)
            if known_claim_ids and any(plan_by_scene.values()) and not graph_scene_ids:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.scene_claims",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="ScenePlan 已声明数学断言，但教学图谱没有场景分配。",
                        fix_instruction="为每个场景登记与 ScenePlan.claim_ids 一致的 scene_claims。",
                    )
                )
            unknown_graph_scenes = graph_scene_ids - set(plan_by_scene)
            if unknown_graph_scenes:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.scene_claims",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="教学图谱 scene_claims 引用了不存在的场景: "
                        + ", ".join(map(str, sorted(unknown_graph_scenes))),
                        fix_instruction="只为实际存在的 Scene 分配断言。",
                    )
                )
            for scene_id, claim_ids in teaching_graph.scene_claims.items():
                graph_claim_set = set(claim_ids)
                unknown = graph_claim_set - known_claim_ids
                missing_from_plan = graph_claim_set - plan_by_scene.get(scene_id, set())
                absent_from_graph = plan_by_scene.get(scene_id, set()) - graph_claim_set
                if unknown:
                    issues.append(
                        PlanCompilerIssue(
                            category="math",
                            field=f"teaching_graph.scene_claims[{scene_id}]",
                            scene_ids=[scene_id]
                            if scene_id in plan_by_scene
                            else [plan.scene_id for plan in plans],
                            message="场景图谱引用了未声明的断言: " + ", ".join(sorted(unknown)),
                            fix_instruction="只引用 LessonSpec 中已有的 claim_id。",
                        )
                    )
                if missing_from_plan or absent_from_graph:
                    mismatch = sorted(missing_from_plan | absent_from_graph)
                    issues.append(
                        PlanCompilerIssue(
                            category="contract",
                            field=f"teaching_graph.scene_claims[{scene_id}]",
                            scene_ids=[scene_id]
                            if scene_id in plan_by_scene
                            else [plan.scene_id for plan in plans],
                            message="教学图谱与 ScenePlan.claim_ids 不一致: " + ", ".join(mismatch),
                            fix_instruction="让 scene_claims 与对应 ScenePlan 的 claim_ids 完全一致。",
                        )
                    )

            for scene_id, plan_claim_ids in plan_by_scene.items():
                if not plan_claim_ids or scene_id in teaching_graph.scene_claims:
                    continue
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field=f"teaching_graph.scene_claims[{scene_id}]",
                        scene_ids=[scene_id],
                        message="ScenePlan 声明了数学断言，但教学图谱没有为该场景分配: "
                        + ", ".join(sorted(plan_claim_ids)),
                        fix_instruction="在 TeachingGraph.scene_claims 中为该场景登记相同的 claim_ids。",
                    )
                )

            if graph_order:
                graph_position = {claim_id: index for index, claim_id in enumerate(graph_order)}
                for claim in lesson_spec.claims:
                    if claim.claim_id not in graph_position:
                        continue
                    for prerequisite in claim.prerequisite_claim_ids:
                        if (
                            prerequisite in graph_position
                            and graph_position[prerequisite] >= graph_position[claim.claim_id]
                        ):
                            issues.append(
                                PlanCompilerIssue(
                                    category="math",
                                    field="teaching_graph.claim_order",
                                    scene_ids=[plan.scene_id for plan in plans],
                                    message=(
                                        f"claim_order 中前置断言 {prerequisite} 未排在依赖断言 "
                                        f"{claim.claim_id} 之前。"
                                    ),
                                    fix_instruction="按从前置知识到结论的顺序重排 claim_order。",
                                )
                            )
                graph_edges = {
                    (edge.prerequisite_claim_id, edge.dependent_claim_id)
                    for edge in teaching_graph.edges
                }
                for claim in lesson_spec.claims:
                    for prerequisite in claim.prerequisite_claim_ids:
                        if (
                            prerequisite in known_claim_ids
                            and (prerequisite, claim.claim_id) not in graph_edges
                        ):
                            issues.append(
                                PlanCompilerIssue(
                                    category="contract",
                                    field="teaching_graph.edges",
                                    scene_ids=[plan.scene_id for plan in plans],
                                    message=(
                                        f"教学图谱缺少 LessonSpec 中的依赖边 {prerequisite} -> "
                                        f"{claim.claim_id}。"
                                    ),
                                    fix_instruction="为每个 prerequisite_claim_ids 补充对应的 TeachingEdge。",
                                )
                            )
            unknown_edges = {
                endpoint
                for edge in teaching_graph.edges
                for endpoint in (edge.prerequisite_claim_id, edge.dependent_claim_id)
                if endpoint not in known_claim_ids
            }
            if unknown_edges:
                issues.append(
                    PlanCompilerIssue(
                        category="contract",
                        field="teaching_graph.edges",
                        scene_ids=[plan.scene_id for plan in plans],
                        message="教学图谱边引用了未声明的断言: " + ", ".join(sorted(unknown_edges)),
                        fix_instruction="删除未知边，或先把对应断言加入 LessonSpec。",
                    )
                )

        first_scene_by_claim: dict[str, int] = {}
        for plan in plans:
            for claim_id in plan.claim_ids:
                first_scene_by_claim.setdefault(claim_id, plan.scene_id)
        for claim in lesson_spec.claims:
            dependent_scene = first_scene_by_claim.get(claim.claim_id)
            if dependent_scene is None:
                continue
            for prerequisite_id in claim.prerequisite_claim_ids:
                prerequisite_scene = first_scene_by_claim.get(prerequisite_id)
                if prerequisite_scene is not None and prerequisite_scene > dependent_scene:
                    issues.append(
                        PlanCompilerIssue(
                            category="math",
                            field=f"claims[{claim.claim_id}].prerequisite_claim_ids",
                            scene_ids=[prerequisite_scene, dependent_scene],
                            message=(
                                f"断言 {claim.claim_id} 在 Scene {dependent_scene} 先于其前置断言 "
                                f"{prerequisite_id}（Scene {prerequisite_scene}）出现。"
                            ),
                            fix_instruction="调整 Scene 分配顺序，先展示所有前置断言再展示依赖断言。",
                        )
                    )

        transition = effective_transition_duration(plan.duration_seconds for plan in plans)
        planned_duration = sum(plan.duration_seconds for plan in plans)
        final_duration = planned_duration - transition * (len(plans) - 1)
        minimum = lesson_spec.requested_duration_min_seconds
        maximum = lesson_spec.requested_duration_max_seconds
        if minimum is not None and final_duration < minimum - 0.05:
            issues.append(
                PlanCompilerIssue(
                    category="timing",
                    field="duration_seconds",
                    scene_ids=[plan.scene_id for plan in plans],
                    message=f"考虑转场后的预计成片时长 {final_duration:.2f}s 小于要求下限 {minimum:.2f}s。",
                    fix_instruction="增加场景定格/吸收停顿，使扣除转场后的成片时长达到要求下限。",
                )
            )
        if maximum is not None and final_duration > maximum + 0.05:
            issues.append(
                PlanCompilerIssue(
                    category="timing",
                    field="duration_seconds",
                    scene_ids=[plan.scene_id for plan in plans],
                    message=f"考虑转场后的预计成片时长 {final_duration:.2f}s 超过要求上限 {maximum:.2f}s。",
                    fix_instruction="压缩场景时长或减少不必要场景，确保扣除转场后的成片不超时。",
                )
            )
        if lesson_spec.scene_policy == "single_visual_unit" and len(plans) > 1:
            issues.append(
                PlanCompilerIssue(
                    category="contract",
                    field="scene_policy",
                    scene_ids=[plan.scene_id for plan in plans],
                    message="教学合同要求单一视觉单元，但概要仍包含多个场景。",
                    fix_instruction="将同一画布中的连续动作合并为一个 Scene。",
                )
            )
        return issues

    @staticmethod
    def _compile_boundary(previous: ScenePlan, current: ScenePlan) -> list[PlanCompilerIssue]:
        previous_available = {
            item.element_id
            for item in [*previous.inherited_elements, *previous.new_elements]
            if item.required
            and item.element_id
            not in {removed.element_id for removed in previous.elements_to_remove}
        }
        current_inherited = {item.element_id for item in current.inherited_elements}
        if not current_inherited:
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
