"""Manim 动画对象生命周期的确定性检查。

该模块只做保守的 AST 分析，不执行模型生成的 Python。它把最容易导致
渲染失败的 ``Transform``、``FadeOut``、重复定义和 MathTex 下标访问问题
提前反馈给 Coder；无法从 AST 确认的视觉细节不会被擅自判定为错误。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from kd1_anime.agents.technical_planner import TechnicalSpec

_INTRODUCERS = {
    "AddTextLetterByLetter",
    "AddTextWordByWord",
    "Create",
    "DrawBorderThenFill",
    "FadeIn",
    "GrowArrow",
    "GrowFromCenter",
    "GrowFromEdge",
    "GrowFromPoint",
    "ShowIncreasingSubsets",
    "ShowSubmobjectsOneByOne",
    "SpinInFromNothing",
    "Write",
}
_REMOVERS = {
    "DisappearToPoint",
    "FadeOut",
    "FadeOutToPoint",
    "ShrinkToCenter",
    "Uncreate",
}
_TRANSFORMS = {
    "Transform",
    "ReplacementTransform",
    "TransformMatchingTex",
    "TransformMatchingShapes",
}
_IN_PLACE_ANIMATIONS = {
    "ApplyMethod",
    "ApplyPointwiseFunction",
    "ApplyWave",
    "Circumscribe",
    "Flash",
    "FocusOn",
    "Indicate",
    "MoveToTarget",
    "Restore",
    "ShowPassingFlash",
    "ShowPassingFlashAround",
    "UpdateFromFunc",
    "Wiggle",
}
_CONTAINER_ANIMATIONS = {"AnimationGroup", "LaggedStart", "Succession", "Group"}
_SCENE_SIDE_EFFECTS = {"play", "add", "remove", "clear"}


@dataclass(frozen=True, slots=True)
class LifecycleValidationResult:
    """生命周期检查结果。"""

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AnimationInvocation:
    operation: str
    source_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()


def repair_required_export_alias_lifecycle(
    code: str,
    technical_spec: TechnicalSpec,
) -> tuple[str, tuple[str, ...]]:
    """修复模型把必需导出对象换成 ``*_initial`` 别名的常见生命周期错误。

    代码模型有时会同时生成 ``v1``（连续性导出区中的最终变量）和
    ``v1_initial``，先引入/变换后者，最后再写 ``v1 = v1_initial``。这在
    Python 语义上只是变量重新绑定，Manim 场景里真正 active 的仍是
    ``v1_initial``；后续对 ``v1.animate`` 的操作和场景边界导出都会失效。

    这是一个非常窄的、可证明安全的兼容修复：只有当

    * ``variable`` 是 TechnicalSpec 要求导出的新对象；
    * 存在 ``variable = variable_suffix`` 的直接重绑定；且
    * 该 suffix 变量在重绑定前已经作为动画/``self.add`` 的源出现；

    才把动画引用改回导出变量，并把无效重绑定替换为 ``pass``。定义、
    字符串和注释不会被文本替换；所有最终安全性仍由 AST、连续性合同和
    生命周期校验继续把关。无法确定时原样返回，不替模型臆造生命周期。
    """

    if not code or not technical_spec.export_element_ids:
        return code, ()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, ()
    construct = _construct_node(tree)
    if construct is None:
        return code, ()

    exported_variables = {
        item.variable_name
        for item in technical_spec.objects
        if item.element_id in technical_spec.export_element_ids and item.variable_name
    }
    if not exported_variables:
        return code, ()

    # 只接受能表达“初始/单位/起点对象”的后缀，避免把任意业务别名
    # 自动重写成边界对象。
    alias_pattern = re.compile(r"^(?P<base>[A-Za-z_]\w*)_(?:initial|base|unit|start)$")
    statements = _statement_nodes(construct)
    rebinding_candidates: list[tuple[ast.Assign, str, str]] = []
    for node in statements:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in exported_variables:
            continue
        if not isinstance(node.value, ast.Name) or node.value.id == target.id:
            continue
        match = alias_pattern.fullmatch(node.value.id)
        if match is None or match.group("base") != target.id:
            continue
        rebinding_candidates.append((node, target.id, node.value.id))
    if not rebinding_candidates:
        return code, ()

    def _names_used_as_scene_sources(limit: int) -> set[str]:
        sources: set[str] = set()
        for node in statements:
            if not isinstance(node, ast.Call) or node.lineno >= limit:
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "add"
            ):
                for argument in node.args:
                    sources.update(_root_names(argument))
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "play"
            ):
                for argument in node.args:
                    for invocation in _animation_invocations(argument):
                        sources.update(invocation.source_names)
                        # A transform source is the only object that becomes
                        # active/changes in place. Targets remain snapshots.
        return sources

    # AST 的列偏移是 UTF-8 字节偏移，使用 bytes 编辑，避免中文字符串
    # 出现在同一行时把字符列和字节列混淆。
    source_bytes = code.encode("utf-8")
    lines = code.splitlines(keepends=True)
    line_offsets: list[int] = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))

    def offset(lineno: int, column: int) -> int:
        if lineno < 1 or lineno > len(lines):
            return len(source_bytes)
        return line_offsets[lineno - 1] + column

    def node_range(node: ast.AST) -> tuple[int, int]:
        start = offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end_line = getattr(node, "end_lineno", node.lineno)
        end_col = getattr(node, "end_col_offset", node.col_offset)
        return start, offset(end_line, end_col)

    edits: list[tuple[int, int, bytes]] = []
    repairs: list[str] = []
    replaced_aliases: set[str] = set()
    removed_ranges: list[tuple[int, int]] = []

    for rebind, variable, alias in rebinding_candidates:
        if alias not in _names_used_as_scene_sources(rebind.lineno):
            continue
        start, end = node_range(rebind)
        # 不处理同一行拼接的多个语句；这种情况下替换成 pass 可能改变
        # 其它语句的缩进/分号结构，交给 Coder 的确定性反馈更安全。
        physical_line = lines[rebind.lineno - 1].strip()
        if not physical_line.startswith(f"{variable}"):
            continue
        edits.append((start, end, b"pass"))
        removed_ranges.append((start, end))
        replaced_aliases.add(alias)
        repairs.append(f"将 {variable} 的 active 别名 {alias} 收敛到导出变量")

    if not replaced_aliases:
        return code, ()

    def inside_removed(start: int, end: int) -> bool:
        return any(start >= left and end <= right for left, right in removed_ranges)

    # Alias 定义本身必须保留（只是变成未使用的纯临时对象），其余
    # Mobject/动画引用改为边界变量。这样不会改写字符串或注释。
    alias_definition_ranges: list[tuple[int, int]] = []
    for node in ast.walk(construct):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id in replaced_aliases
            for target in node.targets
        ):
            alias_definition_ranges.append(node_range(node))

    for node in ast.walk(construct):
        if not isinstance(node, ast.Name) or node.id not in replaced_aliases:
            continue
        start, end = node_range(node)
        if inside_removed(start, end) or any(
            start >= left and end <= right for left, right in alias_definition_ranges
        ):
            continue
        edits.append(
            (
                start,
                end,
                next(
                    variable
                    for _, variable, alias in rebinding_candidates
                    if alias == node.id and alias in replaced_aliases
                ).encode("utf-8"),
            )
        )

    # 同一 AST 名称不会重叠；倒序应用能保持前面计算的字节偏移有效。
    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1]), reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return source_bytes.decode("utf-8"), tuple(dict.fromkeys(repairs))


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _root_names(node: ast.AST) -> set[str]:
    """提取 ``formula[0][1]``、``formula.animate.scale`` 的根变量。"""

    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        return _root_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        result: set[str] = set()
        for item in node.elts:
            result.update(_root_names(item))
        return result
    return set()


def _contains_animate(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "animate" for child in ast.walk(node)
    )


def _animation_invocations(node: ast.AST) -> list[_AnimationInvocation]:
    """从 self.play 参数中提取动画操作，忽略普通参数表达式。"""

    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name in _INTRODUCERS or name in _REMOVERS:
            source: set[str] = set()
            arguments = node.args if name == "FadeOut" else node.args[:1]
            for argument in arguments:
                source.update(_root_names(argument))
            return [_AnimationInvocation(name, tuple(sorted(source)))]
        if name in _TRANSFORMS:
            source = _root_names(node.args[0]) if node.args else set()
            target = _root_names(node.args[1]) if len(node.args) > 1 else set()
            return [_AnimationInvocation(name, tuple(sorted(source)), tuple(sorted(target)))]
        if name in _CONTAINER_ANIMATIONS:
            result: list[_AnimationInvocation] = []
            for arg in node.args:
                result.extend(_animation_invocations(arg))
            return result
        if name in _IN_PLACE_ANIMATIONS:
            source_index = 1 if name == "ApplyPointwiseFunction" else 0
            source = (
                _root_names(node.args[source_index]) if len(node.args) > source_index else set()
            )
            return [_AnimationInvocation("animate", tuple(sorted(source)))]
        if _contains_animate(node):
            return [_AnimationInvocation("animate", tuple(sorted(_root_names(node.func))))]
        return []
    if isinstance(node, ast.Attribute) and node.attr == "animate":
        return [_AnimationInvocation("animate", tuple(sorted(_root_names(node.value))))]
    return []


def _construct_node(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "construct":
            return node
    return None


def _statement_nodes(construct: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """按源代码顺序返回需要处理的赋值、循环绑定和 Scene 调用。"""

    nodes: list[ast.AST] = []
    for node in ast.walk(construct):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.For, ast.AsyncFor)) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in {"play", "add", "remove", "clear"}
        ):
            nodes.append(node)
    return sorted(nodes, key=lambda item: (item.lineno, item.col_offset))


def _side_effects_outside_construct(tree: ast.AST, construct: ast.AST) -> list[str]:
    """找出辅助函数中的 Scene 副作用。

    生命周期模拟只对 construct 建立 active 状态；若辅助函数藏有
    self.play/add/remove/clear，静态状态会被绕过。因此生成代码采用保守
    合同：辅助函数只能构造并返回 Mobject，所有 Scene 副作用必须直接位于
    construct()。
    """

    errors: list[str] = []
    seen: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node is construct:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "self"
                and child.func.attr in _SCENE_SIDE_EFFECTS
            ):
                key = (child.lineno, node.name, child.func.attr)
                if key in seen:
                    continue
                seen.add(key)
                errors.append(
                    f"第 {child.lineno} 行辅助函数 {node.name} 调用了 self.{child.func.attr}()；"
                    "Scene 副作用必须直接写在 construct() 中"
                )
    return errors


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _assignment_aliases(node: ast.Assign | ast.AnnAssign) -> dict[str, set[str]]:
    """提取 ``group = VGroup(existing_a, existing_b)`` 的组合别名。"""

    value = node.value
    roots: set[str] = set()
    if isinstance(value, ast.Call) and _call_name(value) in {"Group", "VGroup"}:
        for argument in value.args:
            roots.update(_root_names(argument))
    elif isinstance(value, ast.Name):
        roots.add(value.id)
    if not roots:
        return {}
    return {name: set(roots) for name in _assignment_names(node)}


def validate_animation_lifecycle(
    code: str,
    technical_spec: TechnicalSpec,
    *,
    renderer: str | None = None,
) -> LifecycleValidationResult:
    """以保守规则检查代码中的对象生命周期。"""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return LifecycleValidationResult(False, (f"生命周期检查无法解析 Python: {exc}",))

    construct = _construct_node(tree)
    if construct is None:
        return LifecycleValidationResult(False, ("生命周期检查找不到 construct()",))

    errors.extend(_side_effects_outside_construct(tree, construct))

    object_by_variable = {
        item.variable_name: item for item in technical_spec.objects if item.variable_name
    }
    element_by_variable = {
        item.variable_name: item.element_id for item in technical_spec.objects if item.variable_name
    }
    required_export_variables = {
        item.variable_name
        for item in technical_spec.objects
        if item.element_id in technical_spec.export_element_ids and item.variable_name
    }
    removed_ids = set(technical_spec.removed_element_ids)
    removed_variables = {
        item.variable_name
        for item in technical_spec.objects
        if item.element_id in removed_ids and item.variable_name
    }

    defined: set[str] = set()
    aliases: dict[str, set[str]] = {}
    active: set[str] = {
        item.variable_name
        for item in technical_spec.objects
        if item.initially_active and item.variable_name
    }
    seen_assignments: set[str] = set()

    def mapped(names: set[str]) -> set[str]:
        result: set[str] = set()
        pending = list(names)
        visited: set[str] = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            if name in element_by_variable:
                result.add(name)
            pending.extend(aliases.get(name, ()))
        return result

    def require_defined(names: set[str], location: int, operation: str) -> None:
        missing = names - defined
        if missing:
            errors.append(
                f"第 {location} 行 {operation} 使用了尚未定义的对象: " + ", ".join(sorted(missing))
            )

    for node in _statement_nodes(construct):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            # 循环变量在 Python 中会在第一次迭代前绑定。生成代码中常用
            # ``for formula in formulas: Write(formula)``，忽略这个绑定
            # 会把合法的循环体误报为“Write 使用未定义对象”。这里只
            # 记录变量名，不执行迭代器或循环体。
            loop_names = _root_names(node.target)
            defined.update(loop_names)
            seen_assignments.update(loop_names)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = set(_assignment_names(node))
            redefined = names & active & seen_assignments
            if redefined:
                errors.append(
                    f"第 {node.lineno} 行重定义仍处于 active 的对象: "
                    + ", ".join(sorted(redefined))
                )
            defined.update(names)
            seen_assignments.update(names)
            for alias, roots in _assignment_aliases(node).items():
                aliases[alias] = roots
            continue

        assert isinstance(node, ast.Call)
        method = node.func.attr
        if method == "clear":
            persistent = active & required_export_variables
            if persistent:
                errors.append(
                    f"第 {node.lineno} 行 self.clear() 会清除必须保留的元素: "
                    + ", ".join(sorted(persistent))
                )
            else:
                warnings.append(f"第 {node.lineno} 行使用 self.clear()，请确认不会丢失连续性对象")
            active.clear()
            continue

        if method in {"add", "remove"}:
            names: set[str] = set()
            for arg in node.args:
                names.update(_root_names(arg))
            require_defined(names, node.lineno, f"self.{method}()")
            mapped_names = mapped(names)
            if method == "add":
                active.update(mapped_names)
            else:
                active.difference_update(mapped_names)
            continue

        if method != "play":
            continue
        for argument in node.args:
            for invocation in _animation_invocations(argument):
                source_names = set(invocation.source_names)
                target_names = set(invocation.target_names)
                require_defined(source_names | target_names, node.lineno, invocation.operation)
                source_mapped = mapped(source_names)
                target_mapped = mapped(target_names)
                if invocation.operation in _INTRODUCERS:
                    duplicate = target_mapped & active
                    if duplicate:
                        errors.append(
                            f"第 {node.lineno} 行 {invocation.operation} 重复引入 active 对象: "
                            + ", ".join(sorted(duplicate))
                        )
                    active.update(target_mapped or source_mapped)
                elif invocation.operation in _REMOVERS:
                    missing = source_mapped - active
                    if missing:
                        errors.append(
                            f"第 {node.lineno} 行 {invocation.operation} 作用于未 active 对象: "
                            + ", ".join(sorted(missing))
                        )
                    active.difference_update(source_mapped)
                elif invocation.operation in _TRANSFORMS:
                    missing = source_mapped - active
                    if missing:
                        errors.append(
                            f"第 {node.lineno} 行 {invocation.operation} 的 source 未 active: "
                            + ", ".join(sorted(missing))
                        )
                    if invocation.operation == "ReplacementTransform":
                        active.difference_update(source_mapped)
                        active.update(target_mapped)
                    # Transform 原地修改 source；target 是目标快照，不能在
                    # 后续事件中当成已经加入 Scene 的对象。
                elif invocation.operation == "animate":
                    missing = source_mapped - active
                    if missing:
                        errors.append(
                            f"第 {node.lineno} 行 animate 作用于未 active 对象: "
                            + ", ".join(sorted(missing))
                        )

    missing_exports = required_export_variables - active
    if missing_exports:
        errors.append("场景结束时必须导出的对象不 active: " + ", ".join(sorted(missing_exports)))
    removed_active = active & removed_variables
    if removed_active:
        errors.append("场景结束时已移除对象仍 active: " + ", ".join(sorted(removed_active)))

    for node in ast.walk(construct):
        if not isinstance(node, ast.Subscript):
            continue
        roots = _root_names(node.value)
        for variable in mapped(roots):
            technical_object = object_by_variable.get(variable)
            if technical_object is None or "Tex" not in technical_object.constructor:
                continue
            if variable not in technical_spec.latex.expected_part_counts:
                warnings.append(
                    f"第 {node.lineno} 行直接访问 {variable} 的 MathTex 子对象，"
                    "TechnicalSpec 未提供 expected_part_counts"
                )

    effective_renderer = renderer or technical_spec.renderer
    if effective_renderer == "opengl":
        for node in ast.walk(construct):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "frame"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "camera"
            ):
                errors.append(f"第 {node.lineno} 行 OpenGL 禁止使用 self.camera.frame")

    deduped_errors = tuple(dict.fromkeys(errors))
    deduped_warnings = tuple(dict.fromkeys(warnings))
    return LifecycleValidationResult(not deduped_errors, deduped_errors, deduped_warnings)
