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

# 这些集合只用于识别少量能安全推断参数角色的常见调用，以及兼容修复
# 函数。它们不是技术合同的能力白名单；未知调用会进入 warning。
_INTRODUCERS = {"Create", "Write", "FadeIn"}
_REMOVERS = {"FadeOut", "Uncreate"}
_TRANSFORMS = {"Transform", "ReplacementTransform"}
_IN_PLACE_ANIMATIONS = {
    "ApplyMethod",
    "ApplyPointwiseFunction",
    "Circumscribe",
    "Flash",
    "Indicate",
    "MoveToTarget",
    "Restore",
    "UpdateFromFunc",
    "Wiggle",
}
_CONTAINER_ANIMATIONS = {"AnimationGroup", "LaggedStart", "Succession", "Group"}
_SCENE_SIDE_EFFECTS = {"play", "add", "remove", "clear"}
_EVENT_MARKER_RE = re.compile(
    r"^\s*#\s*KD1_ANIMATION_EVENT:\s*"
    r"(?P<event_id>[A-Za-z_][A-Za-z0-9_.-]{0,99})\s*$"
)
_AUTO_EVENT_PREFIX = "__auto_"


@dataclass(frozen=True, slots=True)
class LifecycleValidationResult:
    """生命周期检查结果。"""

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unknown_animations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AnimationInvocation:
    operation: str
    source_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    unknown: bool = False


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


def repair_required_export_transform_alias_lifecycle(
    code: str,
    technical_spec: TechnicalSpec,
) -> tuple[str, tuple[str, ...]]:
    """收敛“变换后把临时 target 重绑定为导出变量”的生命周期写法。

    生成代码常写成 ``ReplacementTransform(grid, sheared_grid)``，随后
    ``grid = sheared_grid``。这在 Python 中看似更新了引用，但技术合同
    导出的对象身份已经被 ReplacementTransform 移除，且 ``sheared_grid``
    不是合同变量，静态检查会把 ``grid`` 判定为不 active。若能确认临时
    名称确实是某个必需导出变量的 Transform target，则把替换变换收敛为
    原地 ``Transform``，删除重绑定，并把重绑定之后的临时引用改回合同
    变量。所有判断均来自 AST，不执行生成代码。
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
        rebinding_candidates.append((node, target.id, node.value.id))
    if not rebinding_candidates:
        return code, ()

    # 找到“导出变量 -> 临时 target”的变换调用。只接受发生在重绑定前的
    # Transform/ReplacementTransform，避免把普通业务别名误判为场景边界。
    target_transforms: dict[tuple[str, str], list[ast.Call]] = {}
    for node in statements:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "play"
        ):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or _call_name(child) not in {
                "Transform",
                "ReplacementTransform",
            }:
                continue
            if len(child.args) < 2:
                continue
            source_names = _root_names(child.args[0])
            target_names = _root_names(child.args[1])
            for variable in source_names & exported_variables:
                for alias in target_names:
                    target_transforms.setdefault((variable, alias), []).append(child)

    selected: list[tuple[ast.Assign, str, str, ast.Call]] = []
    for rebind, variable, alias in rebinding_candidates:
        calls = [
            call
            for call in target_transforms.get((variable, alias), ())
            if call.lineno < rebind.lineno
        ]
        if calls:
            selected.append((rebind, variable, alias, calls[-1]))
    if not selected:
        return code, ()

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

    def inside_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start >= left and end <= right for left, right in ranges)

    edits: list[tuple[int, int, bytes]] = []
    skipped_ranges: list[tuple[int, int]] = []
    replacements: dict[str, str] = {}
    repairs: list[str] = []
    for rebind, variable, alias, transform in selected:
        # 只转换 ReplacementTransform。原地 Transform 已保留 source 的
        # active 身份，仍需删除重绑定并收敛后续 alias 引用。
        if _call_name(transform) == "ReplacementTransform":
            func_start, func_end = node_range(transform.func)
            edits.append((func_start, func_end, b"Transform"))
        rebind_start, rebind_end = node_range(rebind)
        edits.append((rebind_start, rebind_end, b"pass"))
        skipped_ranges.append((rebind_start, rebind_end))
        replacements[alias] = variable
        repairs.append(f"将 {variable} 的变换 target {alias} 收敛到导出变量")

    # 保留临时 target 的构造定义和变换调用；重绑定之后对它的引用应当
    # 指向已经原地变换完成的导出对象。定义自身可能出现在重绑定之后，
    # 这种不确定写法不自动改写，交给 Coder 修复。
    definition_ranges: list[tuple[int, int]] = []
    for node in ast.walk(construct):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id in replacements for target in node.targets
        ):
            definition_ranges.append(node_range(node))

    for node in ast.walk(construct):
        if not isinstance(node, ast.Name) or node.id not in replacements:
            continue
        start, end = node_range(node)
        if inside_any(start, end, skipped_ranges) or inside_any(start, end, definition_ranges):
            continue
        # 仅改写重绑定之后的使用；变换调用的 target 必须保留临时对象，
        # 否则会变成 Transform(variable, variable)。
        matching_rebinds = [
            item for item in selected if item[2] == node.id and node.lineno > item[0].lineno
        ]
        if not matching_rebinds:
            continue
        variable = matching_rebinds[-1][1]
        edits.append((start, end, variable.encode("utf-8")))

    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1]), reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return source_bytes.decode("utf-8"), tuple(dict.fromkeys(repairs))


def repair_required_export_replacement_lifecycle(
    code: str,
    technical_spec: TechnicalSpec,
) -> tuple[str, tuple[str, ...]]:
    """保留必需导出 source 的身份，避免替换到未声明的临时 target。

    当代码没有写 ``grid = grid_target`` 这样的重绑定时，前一个兼容修复
    没有可处理的赋值，但 ``ReplacementTransform(grid, grid_target)`` 仍会
    把合同要求导出的 ``grid`` 从 Scene 中移除。若 target 不是另一个已
    声明的合同变量，也将其降为原地 ``Transform``；TechnicalSpec 的
    ``transform`` 语义不会让 target 自动成为 active，而导出 source 的
    active 身份必须得到保留。
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

    replacements: list[ast.Call] = []
    for node in _statement_nodes(construct):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "play"
        ):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and _call_name(child) == "ReplacementTransform"
                and len(child.args) >= 2
            ):
                sources = _root_names(child.args[0])
                targets = _root_names(child.args[1])
                if sources & exported_variables and targets:
                    replacements.append(child)
    if not replacements:
        return code, ()

    source_bytes = code.encode("utf-8")
    lines = code.splitlines(keepends=True)
    line_offsets: list[int] = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))

    def offset(lineno: int, column: int) -> int:
        if lineno < 1 or lineno > len(lines):
            return len(source_bytes)
        return line_offsets[lineno - 1] + column

    edits: list[tuple[int, int, bytes]] = []
    for call in replacements:
        start = offset(call.func.lineno, call.func.col_offset)  # type: ignore[attr-defined]
        end = offset(
            getattr(call.func, "end_lineno", call.func.lineno),
            getattr(call.func, "end_col_offset", call.func.col_offset),
        )
        edits.append((start, end, b"Transform"))
    for start, end, replacement in sorted(edits, reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return (
        source_bytes.decode("utf-8"),
        (
            "将必需导出对象到未声明 target 的 ReplacementTransform 降为 Transform: "
            + ", ".join(
                sorted(
                    {
                        variable
                        for call in replacements
                        for variable in _root_names(call.args[0]) & exported_variables
                    }
                )
            ),
        ),
    )


def repair_removed_active_lifecycle(
    code: str,
    technical_spec: TechnicalSpec,
    errors: tuple[str, ...] | list[str],
) -> tuple[str, tuple[str, ...]]:
    """为明确报告为 active 的移除对象补一条最小 FadeOut。

    ``elements_to_remove`` 是结构化边界合同。Coder 有时能正确实现主体
    动画，却因为阅读了互相矛盾的自然语言 transition_out，漏掉最后的
    FadeOut。此时把场景判死没有必要：生命周期错误已经精确指出了应退出
    的合同变量，可以在 construct() 末尾补一条退出动画，再由完整校验链
    复核。只有错误文本明确列出“已移除对象仍 active”的变量才会触发，
    不会为普通运行时错误或未确认的对象擅自添加动画。
    """

    if not code or not technical_spec.removed_element_ids:
        return code, ()
    reported: set[str] = set()
    marker = "场景结束时已移除对象仍 active:"
    for error in errors:
        text = str(error)
        if marker not in text:
            continue
        reported.update(item.strip() for item in text.split(marker, 1)[1].split(","))
    if not reported:
        return code, ()

    removed_variables = {
        item.variable_name
        for item in technical_spec.objects
        if item.element_id in set(technical_spec.removed_element_ids) and item.variable_name
    }
    variables = sorted(reported & removed_variables)
    if not variables:
        return code, ()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, ()
    construct = _construct_node(tree)
    if construct is None or not construct.body:
        return code, ()
    defined = {
        name
        for node in _statement_nodes(construct)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for name in _assignment_names(node)
    }
    variables = [name for name in variables if name in defined]
    if not variables:
        return code, ()

    lines = code.splitlines(keepends=True)
    line_offsets: list[int] = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))
    last_body = construct.body[-1]
    last_line = int(getattr(last_body, "end_lineno", last_body.lineno))
    insert_at = (
        line_offsets[last_line] if last_line < len(line_offsets) else len(code.encode("utf-8"))
    )
    first_body_line = lines[construct.body[0].lineno - 1] if lines else ""
    indentation = first_body_line[: len(first_body_line) - len(first_body_line.lstrip())]
    cleanup = (
        f"{indentation}# KD1_ANIMATION_EVENT: __auto_remove_active\n"
        f"{indentation}self.play(FadeOut({', '.join(variables)}), run_time=0.5)\n"
    )
    source_bytes = code.encode("utf-8")
    if insert_at == 0 or (insert_at > 0 and source_bytes[insert_at - 1 : insert_at] != b"\n"):
        cleanup = "\n" + cleanup
    source_bytes = source_bytes[:insert_at] + cleanup.encode("utf-8") + source_bytes[insert_at:]
    return (
        source_bytes.decode("utf-8"),
        ("为仍 active 的移除对象补齐 FadeOut: " + ", ".join(variables),),
    )


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


def _is_self_camera_path(node: ast.AST) -> bool:
    """判断属性链是否从 ``self.camera`` 开始。"""

    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return (
        isinstance(current, ast.Name)
        and current.id == "self"
        and isinstance(node, ast.Attribute)
        and (node.attr == "camera" or _is_self_camera_path(node.value))
    )


def _animate_source_names(node: ast.AST) -> set[str]:
    """提取 Mobject.animate 的根变量，忽略 ThreeDScene 相机运镜。"""

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "animate":
            if _is_self_camera_path(child.value):
                return set()
            return _root_names(child.value)
    return _root_names(node)


def _contains_camera_animate(node: ast.AST) -> bool:
    """判断一段动画表达式是否只是在驱动 Scene 相机。"""

    return any(
        isinstance(child, ast.Attribute)
        and child.attr == "animate"
        and _is_self_camera_path(child.value)
        for child in ast.walk(node)
    )


def _contains_camera_reference(node: ast.AST) -> bool:
    """判断表达式是否引用 ``self.camera``/``self.camera.frame``。"""

    return any(
        isinstance(child, ast.Attribute) and _is_self_camera_path(child) for child in ast.walk(node)
    )


def _event_markers(code: str) -> list[tuple[int, str]]:
    """读取源代码中的语义动画事件标记。"""

    markers: list[tuple[int, str]] = []
    for line_number, line in enumerate(code.splitlines(), start=1):
        match = _EVENT_MARKER_RE.match(line)
        if match:
            markers.append((line_number, match.group("event_id")))
    return markers


def _marker_before_line(lines: list[str], line_number: int) -> str | None:
    """返回 self.play 前最近的事件标记，允许空行和普通注释。"""

    index = line_number - 2
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            match = _EVENT_MARKER_RE.match(lines[index])
            if match:
                return match.group("event_id")
            index -= 1
            continue
        break
    return None


def _animation_invocations(node: ast.AST) -> list[_AnimationInvocation]:
    """从 self.play 参数中提取动画操作，忽略普通参数表达式。"""

    if isinstance(node, ast.Call):
        name = _call_name(node)
        if _contains_camera_reference(node):
            return [_AnimationInvocation("camera")]
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
            source_names = _animate_source_names(node)
            operation = (
                "camera" if not source_names and _contains_camera_animate(node) else "animate"
            )
            return [_AnimationInvocation(operation, tuple(sorted(source_names)))]
        # 未知的动画工厂不应被当成非法 API。调用方会通过语义事件标记
        # 提供状态解释，这里仅提取可能的对象根名并记录 warning。
        source: set[str] = set()
        for argument in node.args:
            source.update(_root_names(argument))
        return [
            _AnimationInvocation(f"unknown:{name or 'call'}", tuple(sorted(source)), unknown=True)
        ]
    if isinstance(node, ast.Attribute) and node.attr == "animate":
        if _is_self_camera_path(node.value):
            return [_AnimationInvocation("camera")]
        return [_AnimationInvocation("animate", tuple(sorted(_root_names(node.value))))]
    if isinstance(node, ast.Name):
        return [_AnimationInvocation(f"unknown:{node.id}", (node.id,), unknown=True)]
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
    """以语义事件合同检查代码中的对象生命周期。

    具体动画类不是稳定的能力边界。代码只需要在每个 ``self.play`` 前写
    ``# KD1_ANIMATION_EVENT: <event_id>``，其状态变化由 TechnicalSpec 的
    ``semantic_action`` 决定；无法识别的动画调用只产生 warning。这样
    新的 Manim 动画可以先用于实验，而不会被旧的名称列表阻断。
    """

    errors: list[str] = []
    warnings: list[str] = []
    unknown_animations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return LifecycleValidationResult(False, (f"生命周期检查无法解析 Python: {exc}",))

    construct = _construct_node(tree)
    if construct is None:
        return LifecycleValidationResult(False, ("生命周期检查找不到 construct()",))

    errors.extend(_side_effects_outside_construct(tree, construct))
    lines = code.splitlines()
    event_by_id = {event.event_id: event for event in technical_spec.animations}
    markers = _event_markers(code)
    marker_ids = [event_id for _, event_id in markers]
    for event_id in sorted({item for item in marker_ids if marker_ids.count(item) > 1}):
        errors.append(f"动画事件标记重复: {event_id}")

    object_by_variable = {
        item.variable_name: item for item in technical_spec.objects if item.variable_name
    }
    element_by_variable = {
        item.variable_name: item.element_id for item in technical_spec.objects if item.variable_name
    }
    variable_by_element = {
        item.element_id: item.variable_name for item in technical_spec.objects if item.variable_name
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
    used_event_ids: set[str] = set()
    markers_required = bool(technical_spec.animations)

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

    def require_defined(names: set[str], location: int, description: str) -> None:
        missing = names - defined
        if missing:
            errors.append(
                f"第 {location} 行 {description} 使用了尚未定义的对象: "
                + ", ".join(sorted(missing))
            )

    def contract_variables(event, field_name: str) -> set[str]:
        ids = getattr(event, field_name)
        return {
            variable_by_element[element_id]
            for element_id in ids
            if element_id in variable_by_element
        }

    for node in _statement_nodes(construct):
        if isinstance(node, (ast.For, ast.AsyncFor)):
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

        marker_id = _marker_before_line(lines, node.lineno)
        event = event_by_id.get(marker_id or "")
        if marker_id is None and markers_required:
            errors.append(
                f"第 {node.lineno} 行 self.play() 缺少语义事件标记；"
                "请在上一行写 # KD1_ANIMATION_EVENT: <event_id>"
            )
        elif marker_id is not None:
            if marker_id in used_event_ids:
                errors.append(f"第 {node.lineno} 行重复使用动画事件标记: {marker_id}")
            used_event_ids.add(marker_id)
            if event is None and not marker_id.startswith(_AUTO_EVENT_PREFIX):
                errors.append(
                    f"第 {node.lineno} 行动画事件标记未在 TechnicalSpec 中声明: {marker_id}"
                )

        invocations: list[_AnimationInvocation] = []
        for argument in node.args:
            invocations.extend(_animation_invocations(argument))
        if not invocations and node.args:
            invocations.append(_AnimationInvocation("unknown:expression", unknown=True))

        for invocation in invocations:
            source_names = set(invocation.source_names)
            target_names = set(invocation.target_names)
            if invocation.operation == "camera":
                continue
            require_defined(
                source_names | target_names,
                node.lineno,
                invocation.operation,
            )
            if invocation.unknown:
                detail = f"第 {node.lineno} 行 {invocation.operation}"
                unknown_animations.append(detail)
                warnings.append(
                    f"[unknown-animation] {detail} 未被生命周期分析器识别，"
                    f"按事件 {marker_id or '未标记'} 的语义合同继续"
                )

            # 对少量能可靠识别参数角色的调用保留安全检查；未知调用
            # 不猜测 source/target，不因新动画名称而误报。
            if not invocation.unknown and invocation.operation in _REMOVERS | _TRANSFORMS | {
                "animate"
            }:
                source_mapped = mapped(source_names)
                missing = source_mapped - active
                if missing:
                    if invocation.operation == "animate":
                        message = f"第 {node.lineno} 行 animate 作用于未 active 对象: "
                    else:
                        message = (
                            f"第 {node.lineno} 行 {invocation.operation} 的 source 未 active: "
                        )
                    errors.append(message + ", ".join(sorted(missing)))

        if event is None:
            # 仅用于没有技术事件的独立生命周期检查，或窄范围自动修复。
            # 有事件的正式候选在上面已经报告未声明 marker；不再猜测状态。
            if marker_id is None and not markers_required:
                for invocation in invocations:
                    source_mapped = mapped(set(invocation.source_names))
                    target_mapped = mapped(set(invocation.target_names))
                    if invocation.operation in _INTRODUCERS:
                        active.update(target_mapped or source_mapped)
                    elif invocation.operation in _REMOVERS:
                        active.difference_update(source_mapped)
                    elif invocation.operation == "ReplacementTransform":
                        active.difference_update(source_mapped)
                        active.update(target_mapped)
                    elif invocation.unknown:
                        # 没有技术事件时无法判断未知动画是入场还是原地
                        # 更新；保守地把它的参数视为可能被引入的对象。
                        active.update(source_mapped)
            elif marker_id is not None and marker_id.startswith(_AUTO_EVENT_PREFIX):
                # 后备代码/窄范围自动修复使用的 marker 不属于技术计划，
                # 但其状态变化仍按实际参数做最小、可验证的模拟。
                auto_sources = mapped(
                    {name for invocation in invocations for name in invocation.source_names}
                )
                if marker_id.startswith("__auto_remove"):
                    missing = auto_sources - active
                    if missing:
                        errors.append(
                            f"第 {node.lineno} 行自动 remove 作用于未 active 对象: "
                            + ", ".join(sorted(missing))
                        )
                    active.difference_update(auto_sources)
                elif marker_id.startswith("__auto_introduce"):
                    active.update(auto_sources)
                elif marker_id.startswith("__auto_update"):
                    missing = auto_sources - active
                    if missing:
                        errors.append(
                            f"第 {node.lineno} 行自动 update 作用于未 active 对象: "
                            + ", ".join(sorted(missing))
                        )
            continue

        action = event.semantic_action
        event_sources = contract_variables(event, "source_element_ids")
        event_targets = contract_variables(event, "target_element_ids")
        event_creates = contract_variables(event, "create_element_ids")
        event_removes = contract_variables(event, "remove_element_ids")
        actual_sources = mapped({name for call in invocations for name in call.source_names})
        actual_targets = mapped({name for call in invocations for name in call.target_names})
        expected = event_sources | event_targets | event_creates | event_removes
        if expected - (actual_sources | actual_targets):
            errors.append(
                f"第 {node.lineno} 行事件 {event.event_id} 未操作合同对象: "
                + ", ".join(sorted(expected - (actual_sources | actual_targets)))
            )
        actual_exits = mapped(
            {
                name
                for call in invocations
                if call.operation in _REMOVERS or call.operation == "ReplacementTransform"
                for name in call.source_names
            }
        )
        if actual_exits and action != "remove":
            errors.append(f"第 {node.lineno} 行实际退出对象与 {action} 语义不符")
            active.difference_update(actual_exits)
        if action == "introduce":
            missing = event_sources & active
            if missing:
                errors.append(
                    f"第 {node.lineno} 行 introduce source 已 active: " + ", ".join(sorted(missing))
                )
            introduced = event_targets | event_creates
            if introduced - defined:
                require_defined(introduced - defined, node.lineno, "introduce")
            active.update(introduced & (actual_sources | actual_targets))
        elif action == "update":
            missing = event_sources - active
            if missing:
                errors.append(
                    f"第 {node.lineno} 行 update source 未 active: " + ", ".join(sorted(missing))
                )
            if event_creates:
                errors.append(
                    f"第 {node.lineno} 行 update 不能引入对象: " + ", ".join(sorted(event_creates))
                )
            if event_removes:
                errors.append(
                    f"第 {node.lineno} 行 update 不能移除对象: " + ", ".join(sorted(event_removes))
                )
        elif action == "remove":
            exit_names = event_sources | event_removes
            missing = exit_names - active
            if missing:
                errors.append(
                    f"第 {node.lineno} 行 remove 作用于未 active 对象: "
                    + ", ".join(sorted(missing))
                )
            active.difference_update(exit_names)
        elif action == "hold":
            missing = event_sources - active
            if missing:
                errors.append(
                    f"第 {node.lineno} 行 hold source 未 active: " + ", ".join(sorted(missing))
                )
        elif action == "camera":
            # 相机事件不改变 Mobject 状态。
            pass

    if markers_required:
        unused = set(event_by_id) - used_event_ids
        if unused:
            warnings.append(
                "TechnicalSpec 中没有在代码中找到对应 marker 的事件: " + ", ".join(sorted(unused))
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
    return LifecycleValidationResult(
        not deduped_errors,
        deduped_errors,
        deduped_warnings,
        tuple(dict.fromkeys(unknown_animations)),
    )


def detect_unknown_animations(
    code: str,
    technical_spec: TechnicalSpec,
    *,
    renderer: str | None = None,
) -> tuple[str, ...]:
    """返回无法由静态分析器识别的动画调用位置。

    这是诊断信息，不代表代码无效；调用方可据此开启额外 Smoke Render。
    """

    return validate_animation_lifecycle(code, technical_spec, renderer=renderer).unknown_animations


__all__ = [
    "LifecycleValidationResult",
    "detect_unknown_animations",
    "repair_removed_active_lifecycle",
    "repair_required_export_alias_lifecycle",
    "repair_required_export_replacement_lifecycle",
    "repair_required_export_transform_alias_lifecycle",
    "validate_animation_lifecycle",
]
