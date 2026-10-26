"""Compile explicit scene contracts without interpreting arbitrary Python."""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import math

from kd1_anime.domain.planning import ScenePlan
from kd1_anime.domain.program import SceneProgram
from kd1_anime.manim_catalog_0201 import CATALOG
from kd1_anime.models import validated_copy


class ProgramCompileError(ValueError):
    pass


def expression_source(expression: str) -> str:
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ProgramCompileError("Invalid function expression") from exc
    if sum(1 for _ in ast.walk(root)) > 100:
        raise ProgramCompileError("Function expression is too complex")

    def emit(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
            and math.isfinite(node.value)
            and abs(node.value) <= 1e6
        ):
            return repr(node.value)
        if isinstance(node, ast.Name) and node.id in {"x", "pi", "e"}:
            return node.id if node.id == "x" else "np." + node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ("+" if isinstance(node.op, ast.UAdd) else "-") + "(" + emit(node.operand) + ")"
        if isinstance(node, ast.BinOp) and type(node.op) in {
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
        }:
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[
                type(node.op)
            ]
            return "(" + emit(node.left) + op + emit(node.right) + ")"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"sin", "cos", "tan", "exp", "log", "sqrt", "abs"}
            and len(node.args) == 1
            and not node.keywords
        ):
            return "np." + node.func.id + "(" + emit(node.args[0]) + ")"
        raise ProgramCompileError("Unsupported function expression node")

    return emit(root.body)


def compile_program(program: SceneProgram, plan: ScenePlan) -> str:
    program = validated_copy(program)
    plan = validated_copy(plan)
    if (
        program.scene_id != plan.scene_id
        or abs(program.duration_seconds - plan.duration_seconds) > 1e-6
    ):
        raise ProgramCompileError("Program identity/duration differs from the approved plan")
    objects = {obj.element_id: obj for obj in program.objects}
    variables = {obj.variable_name for obj in program.objects}
    if len(objects) != len(program.objects) or len(variables) != len(program.objects):
        raise ProgramCompileError("Duplicate object identity or variable")
    reserved = {"self", "COLORS", "tex_template", "np", *CATALOG["symbols"]}
    reserved.update(dir(builtins))
    if any(obj.exported and obj.variable_name.startswith("_kd1_") for obj in program.objects):
        raise ProgramCompileError("Export name conflicts with a compiler reserved name")
    if program.class_name in reserved or program.class_name.startswith("_kd1_"):
        raise ProgramCompileError("Class name shadows a runtime symbol")
    reserved.add(program.class_name)
    renamed = []
    for index, original in enumerate(program.objects):
        obj = original
        if obj.variable_name in reserved and obj.exported:
            raise ProgramCompileError("Export name shadows a runtime symbol")
        if not obj.exported:
            obj = validated_copy(obj, update={"variable_name": f"_kd1_object_{index}"})
        renamed.append(obj)
    program = validated_copy(program, update={"objects": renamed})
    objects = {obj.element_id: obj for obj in program.objects}
    variables = {obj.variable_name for obj in program.objects}
    if any(
        keyword.iskeyword(name) or name.startswith("__") or name in reserved for name in variables
    ) or keyword.iskeyword(program.class_name):
        raise ProgramCompileError("Unsafe Python identifier")
    removed = {obj.element_id for obj in plan.elements_to_remove}
    required = {
        obj.element_id
        for obj in (*plan.inherited_elements, *plan.new_elements)
        if obj.required and obj.element_id not in removed
    }
    exported = {obj.element_id for obj in program.objects if obj.exported}
    if exported != required:
        raise ProgramCompileError("Program exports differ from the approved boundary")
    if len({event.event_id for event in program.events}) != len(program.events):
        raise ProgramCompileError("Duplicate event identity")
    lines = [
        "from manim import *",
        "import numpy as np",
        "",
        f"class {program.class_name}(Scene):",
        "    def construct(self):",
    ]

    def q(value):
        return json.dumps(value, ensure_ascii=False)

    colors = plan.global_visual_state.colors
    lines.append("        COLORS = " + q(colors))
    lines.append("        self.camera.background_color = " + q(plan.global_visual_state.background))
    if any(obj.kind in {"Tex", "MathTex"} for obj in program.objects):
        lines.extend(
            [
                '        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")',
                '        tex_template.add_to_preamble(r"\\usepackage{amsmath}")',
            ]
        )

    if any(obj.kind in {"Tex", "MathTex"} for obj in program.objects):
        lines.append('        tex_template.add_to_preamble(r"\\usepackage{ctex}")')
        lines.append("        config.tex_template = tex_template")

    def constructor(obj):
        color = f"COLORS[{q(obj.color_key)}]"
        if obj.color_key not in colors:
            raise ProgramCompileError("Missing global color")
        if obj.kind in {"Text", "Tex", "MathTex"}:
            extra = (
                ", tex_template=tex_template"
                if obj.kind != "Text"
                else ", font=" + q(plan.global_visual_state.fonts["text"])
            )
            text = f"{obj.kind}({q(obj.text)}, font_size={obj.font_size!r}, color={color}{extra})"
        elif obj.kind == "Circle":
            text = f"Circle(radius={obj.radius!r}, color={color})"
        elif obj.kind == "Rectangle":
            text = f"Rectangle(width={obj.width!r}, height={obj.height!r}, color={color})"
        elif obj.kind == "Square":
            text = f"Square(side_length={obj.side_length!r}, color={color})"
        elif obj.kind == "Polygon":
            text = f"Polygon({', '.join(q(list(point)) for point in obj.vertices)}, color={color})"
        elif obj.kind in {"Line", "Arrow"}:
            if obj.start == obj.end:
                raise ProgramCompileError("Degenerate line")
            text = f"{obj.kind}({q(list(obj.start))}, {q(list(obj.end))}, color={color}, buff=0)"
        elif obj.kind == "Dot":
            text = f"Dot(radius={obj.radius!r}, color={color})"
        elif obj.kind == "Axes":
            if obj.x_range[0] >= obj.x_range[1] or obj.y_range[0] >= obj.y_range[1]:
                raise ProgramCompileError("Invalid axis range")
            text = f"Axes(x_range={q(list(obj.x_range))}, y_range={q(list(obj.y_range))}, x_length={obj.x_length!r}, y_length={obj.y_length!r}, axis_config={{'color': {color}}})"
        elif obj.kind == "Graph":
            axis = objects.get(obj.axes_element_id)
            if axis is None or axis.kind != "Axes" or axis.variable_name not in defined:
                raise ProgramCompileError("Graph needs a previously defined Axes")
            if obj.exported and not axis.exported:
                raise ProgramCompileError("Exported graph requires exported axes")
            if obj.x_range[0] >= obj.x_range[1] or obj.position != (0, 0, 0):
                raise ProgramCompileError("Graph needs an increasing domain and its axes' position")
            return f"{axis.variable_name}.plot(lambda x: {expression_source(obj.expression)}, x_range={q(list(obj.x_range))}, color={color})"
        else:
            raise ProgramCompileError("Unsupported object")
        placement = "shift" if obj.kind in {"Polygon", "Line", "Arrow"} else "move_to"
        return text + f".{placement}({q(list(obj.position))})"

    defined: set[str] = set()
    # Boundary declarations must be self contained and precede internal objects.
    for is_exported in (True, False):
        if is_exported:
            lines.append("        # KD1_CONTINUITY_EXPORT_BEGIN")
        for obj in program.objects:
            if obj.exported != is_exported:
                continue
            if is_exported:
                lines.append("        # element_id: " + obj.element_id)
            lines.append(f"        {obj.variable_name} = {constructor(obj)}")
            defined.add(obj.variable_name)
        if is_exported:
            lines.append("        # KD1_CONTINUITY_EXPORT_END")
    active = {obj.element_id for obj in program.objects if obj.initially_active}
    if active:
        lines.append(
            "        self.add("
            + ", ".join(objects[key].variable_name for key in sorted(active))
            + ")"
        )
    elapsed = 0.0
    for event in program.events:
        if (
            event.start_seconds < elapsed - 1e-6
            or event.end_seconds > program.duration_seconds + 1e-6
        ):
            raise ProgramCompileError(
                "Overlapping, unordered or out-of-range events are unsupported"
            )
        if event.start_seconds > elapsed:
            lines.append(f"        self.wait({event.start_seconds - elapsed!r})")
        lines.append("        # KD1_ANIMATION_EVENT: " + event.event_id)
        duration = event.end_seconds - event.start_seconds
        if any(key not in objects for key in (*event.source_ids, *event.target_ids)):
            raise ProgramCompileError("Unknown event object")
        names = [objects[key].variable_name for key in event.source_ids]
        op = event.operation
        if event.target_ids and op not in {"transform", "replacement_transform"}:
            raise ProgramCompileError("Unexpected animation targets")
        if event.vector is not None and op not in {"move_to", "shift"}:
            raise ProgramCompileError("Unexpected vector parameter")
        if event.factor is not None and op != "scale":
            raise ProgramCompileError("Unexpected scale parameter")
        if event.angle is not None and op != "rotate":
            raise ProgramCompileError("Unexpected angle parameter")
        if op == "wait":
            if names or event.target_ids:
                raise ProgramCompileError("Wait cannot reference objects")
            lines.append(f"        self.wait({duration!r})")
        elif not names or len(set(event.source_ids)) != len(names):
            raise ProgramCompileError("Animation needs unique source objects")
        elif op in {"add", "create", "write", "fade_in"}:
            if active.intersection(event.source_ids):
                raise ProgramCompileError("Object introduced twice")
            if op == "add":
                lines.extend(
                    [f"        self.add({', '.join(names)})", f"        self.wait({duration!r})"]
                )
            else:
                call = {"create": "Create", "write": "Write", "fade_in": "FadeIn"}[op]
                lines.append(
                    f"        self.play({', '.join(f'{call}({name})' for name in names)}, run_time={duration!r})"
                )
            active.update(event.source_ids)
        else:
            if not set(event.source_ids).issubset(active):
                raise ProgramCompileError("Animation uses inactive objects")
            if op == "fade_out":
                calls = [f"FadeOut({name})" for name in names]
                active.difference_update(event.source_ids)
            elif op in {"transform", "replacement_transform"}:
                if len(names) != 1 or len(event.target_ids) != 1 or event.target_ids[0] in active:
                    raise ProgramCompileError(
                        "Transform needs one active source and inactive target"
                    )
                if op == "transform" and event.source_ids[0] in exported:
                    raise ProgramCompileError(
                        "Mutated exported objects require an explicit replacement boundary"
                    )
                target = objects[event.target_ids[0]].variable_name
                call = "Transform" if op == "transform" else "ReplacementTransform"
                calls = [f"{call}({names[0]}, {target})"]
                if op == "replacement_transform":
                    active.difference_update(event.source_ids)
                    active.update(event.target_ids)
            elif op in {"move_to", "shift", "scale", "rotate"}:
                if exported.intersection(event.source_ids):
                    raise ProgramCompileError(
                        "Mutated exported objects need a complete final boundary"
                    )
                value = (
                    event.vector
                    if op in {"move_to", "shift"}
                    else event.factor
                    if op == "scale"
                    else event.angle
                )
                if value is None:
                    raise ProgramCompileError("Missing animation parameter")
                argument = q(list(value)) if isinstance(value, tuple) else repr(value)
                calls = [f"{name}.animate.{op}({argument})" for name in names]
            else:
                raise ProgramCompileError("Unsupported event")
            lines.append(f"        self.play({', '.join(calls)}, run_time={duration!r})")
        elapsed = event.end_seconds
    if not exported.issubset(active) or removed.intersection(active):
        raise ProgramCompileError("Final object lifecycle differs from boundary contract")
    if elapsed < program.duration_seconds:
        lines.append(f"        self.wait({program.duration_seconds - elapsed!r})")
    return "\n".join(lines) + "\n"
